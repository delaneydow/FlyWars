#mirror_planner.py

# GOAL: 
# - integrate with mirror
# - take aim XY from planner
# - account for latency
# - produce safe (u,v) mirror commands

import numpy as np
from matplotlib.path import Path as MplPath
import serial # for access with mre-3 serial port 
import time


class MirrorPlanner: 
    def __init__(self, port="/dev/ttyUSB0", baud=115200, map_file="mirror_map1.npz", spot_radius_px=None):

        # serial compatibility
        self.ser = serial.Serial(port, baud, timeout=0.01)

        # viable mirror points
        data = np.load(map_file)

        # calibrations samples
        self.uv = data["uv"]
        self.xy = data["xy"]

        # lookup grid
        self.x_map = data["x_map"]
        self.y_map = data["y_map"]
        self.u_map = data["u_map"]
        self.v_map = data["v_map"]

        #normalization
        self.uv_mean = data["uv_mean"]
        self.uv_std = data["uv_std"]

        # command bounds
        self.u_min, self.u_max, self.v_min, self.v_max = data["bounds"]

        self.hull_path = MplPath(data["hull"]) #convex hull for mirror reachability
        self.spot_radius_px = spot_radius_px or 6.0 #TODO figure out how to integrate this again

    def clamp_uv(self, u, v): 
        return(
            np.clip(u, self.u_min, self.u_max),
            np.clip(v, self.v_min, self.v_max)
        )
    
    def is_reachable(self, x, y): # checks if coordinates can be accessed
        return self.hull_path.contains_point((x,y))
    
    # clamp aim points using spot radius if needed
    def clip_to_reachable(self, x, y): 
        # removes horns, extrapolation instability, mirror overdrive
        if self.hull_path.contains_point((x,y)): 
            return x, y
        
        # snap to nearest reachable beam point
        d = np.hypot(self.x_map - x, self.y_map - y)
        idx = np.argmin(d)

        return self.x_map[idx], self.y_map[idx]
        
    def predict_xy(self, u, v): 
        uvn = (np.column_stack([u, v]) - self.uv_mean) / self.uv_std
        x = self.fx(uvn[:,0], uvn[:,1])
        y = self.fy(uvn[:,0], uvn[:,1])
        return np.column_stack([x,y])
        
        #TODO go back & see if this can be faster via pre-computed UV --> XY map
    def find_uv_for_xy(self, x_target, y_target): 

        # clip first
        x_target, y_target = self.clip_to_reachable(x_target, y_target)

        d = np.hypot(self.x_map -x_target,
                     self.y_map - y_target)
        idx = np.argmin(d)

        return (
            np.clip(self.u_map[idx], self.u_min, self.u_max),
            np.clip(self.v_map[idx], self.v_min, self.v_max)
        )
    
    def send_uv(self, u, v): 

        " Send U/V coordinates as X/Y commands to MRE-3 via USB serial "
        "MRE-3 Expects X and Y within range -1.0 to 1.0 (already calibrated for)"

        # send commands 
        self.ser.write(u.encode())
        #time.sleep(0.01) TODO try without to minimize latency 
        self.ser.write(v.encode())
        #time.sleep(0.01)

        # read echo or status 
        if self.ser.in_waiting:
            resp = self.ser.read(self.ser.in_waiting)
            print("[Mirror] response:", resp.decode(errors='ignore'))
        
        
        cmd = f"{u:.3f},{v:.3f}\n"
        self.ser.write(cmd.encode())