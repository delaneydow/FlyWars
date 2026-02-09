#mirror_planner.py

# GOAL: 
# - integrate with mirror
# - take aim XY from planner
# - account for latency
# - produce safe (u,v) mirror commands

import numpy as np
from matplotlib.path import Path
import pyserial # for access with mre-3 serial port 


class MirrorPlanner: 
    def __init__(self, map_file="mirror_map.npz"):

        data = np.load(map_file)

        self.u = data["u"]
        self.v = data["y"]
        self.x = data["x"]
        self.y = data[y]
        self.u_min, self.u_max, self.v_min, self.v_max = data["bounds"]
        self.hull_path = Path(data["hull"])
        
        #self.spot_radius_px = spot_radius_px #TODO figure out how to integrate this again

    def clamp_uv(self, u, v): 
        return(
            np.clip(u, self.u_min, self.u_max),
            np.clip(v, self.v_min, self.v_max)
        )
    
    def is_reachable(self, x, y): # checks if coordinates can be accessed
        return self.hull_path.contains_point((x,y))
        
    def predict_xy(self, u, v): 
        uvn = (np.column_stack([u, v]) - self.uv_mean) / self.uv_std
        x = self.fx(uvn[:,0], uvn[:,1])
        y = self.fy(uvn[:,0], uvn[:,1])
        return np.column_stack([x,y])
        
        #TODO go back & see if this can be faster via pre-computed UV --> XY map
    def find_uv_for_xy(self, x_target, y_target): 

        d = np.hypot(self.x, x_target, self.y - y_target)
        idx = np.argmin(d)

        return (
            np.clip(self.u[idx], self.u_min, self.u_max),
            np.clip(self.v[idx], self.v_min, self.v_max)
        )