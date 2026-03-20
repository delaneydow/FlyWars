#mirror_planner.py

# GOAL: 
# - integrate with mirror
# - take aim XY from planner
# - account for latency
# - produce safe (u,v) mirror commands

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree #make mirror path / calibration faster 
import serial # for access with mre-3 serial port 
import time

#MRE-3 settling spec anchor points
_SETTLE_DEG_SMALL = 0.1
_SETTLE_MS_SMALL = 2.0
_SETTLE_DEG_LARGE = 10.0
_SETTLE_MS_LARGE = 8.0
_DEG_PER_UNIT = 25.0

_SETTLE_SLOPE = (_SETTLE_MS_LARGE - _SETTLE_MS_SMALL) / (_SETTLE_DEG_LARGE - _SETTLE_DEG_SMALL)

def _settle_time_s(delta_u: float, delta_v: float) -> float: 
    "Compute settle time from move magnitude in input units"
    delta_deg = np.hypot(delta_u, delta_v) * _DEG_PER_UNIT
    delta_deg = np.clip(delta_deg, _SETTLE_DEG_SMALL, _SETTLE_DEG_LARGE)
    ms = _SETTLE_MS_SMALL+ _SETTLE_SLOPE * (delta_deg - _SETTLE_DEG_SMALL)
    return max(ms, _SETTLE_FLOOR_MS) / 1000.0


class MirrorPlanner: 
    def __init__(self, port="/dev/serial/by-id/usb-Optotune_Virtual_ComPort_3578335B3233-if00", baud=115200, map_file="mirror_map1.npz", spot_radius_px=None):

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

        self._xy_tree = cKDTree(np.column_stack([self.x_map, self.y_map]))
        
        # trackk= current mirror position for delta-based settle calc
        self._current_uv = (0.0, 0.0) #assume centered at init

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

        # precomputed KDTree query --> O log n runtime 
        _, idx = self._xy_tree.query([x_target, y_target])
        return (
            float(np.clip(self.u_map[idx], self.u_min, self.u_max)),
            float(np.clip(self.v_map[idx], self.v_min, self.v_max))
        )   

    
    def send_uv(self, u, v): 

        " Send U/V coordinates as X/Y commands to MRE-3 via USB serial "
        "MRE-3 Expects X and Y within range -1.0 to 1.0 (already calibrated for)"
        "block until mirror has settled"

        # Clamp values just to be safe
        u = float(np.clip(u, -1.0, 1.0))
        v = float(np.clip(v, -1.0, 1.0))

        delta_u = u - self._current_uv[0]
        delta_v = v - self._current_uv[1]
        settle = _settle_time_s(delta_u, delta_v)

        # Send to serial port as one command
        self.ser.write(f"XY={u:.3f};{v:.3f}\r\n".encode())

        # Optional: read echo / status
        if self.ser.in_waiting:
            resp = self.ser.read(self.ser.in_waiting)
            #print("[Mirror response]", resp.decode(errors='ignore'))

        self._current_uv = (u, v)
        time.sleep(settle)
    
    def off(self): 
        #print("[MIRROR] CENTER + STOP")
        try: 
            self.send_uv(0.0, 0.0) #use existing module to send command
        except Exception:
            pass

    def close(self): 
        self.ser.close() 