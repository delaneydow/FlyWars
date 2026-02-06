#mirror_planner.py

# GOAL: 
# - integrate with mirror
# - take aim XY from planner
# - account for latency
# - produce safe (u,v) mirror commands

import numpy as np

class MirrorPlanner: 
    def __init__(self, fx, fy, uv_mean, uv_std, uv_bounds, spot_radius_px):
        self.fx = fx
        self.fy = fy
        self.uv_mean = uv_mean
        self.uv_std = uv_std
        self.u_min, self.u_max, self.v_min, self.v_max = uv_bounds
        self.spot_radius_px = spot_radius_px

        def clamp_uv(self, u, v): 
            return(
                np.clip(u, self.u_min, self.u_max),
                np.clip(v, self.v_min, self.v_max)
            )
        
        def predict_xy(self, u, v): 
            uvn = (np.column_stack([u, v]) - self.uv_mean) / self.uv_std
            x = self.fx(uvn[:,0], uvn[:,1])
            y = self.fy(uvn[:,0], uvn[:,1])
            return np.column_stack([x,y])
        
        #TODO go back & see if this can be faster
        def find_uv_for_xy(self, x_target, y_target, grid=40): 
            u = np.linspace(self.u_min, self.u_max, grid)
            v = np.linspace(self.v_min, self.v_max, grid)
            uu, vv = np.meshgrid(u,v)

            uv = np.column_stack([uu.ravel(), vv.ravel()])
            xy = self.predict_xy(uv[:,0], uv[:,1])

            d = np.linalg.norm(xy - np.array([x_target, y_target]), axis=1)

            best = np.argmin(d)

            return self.clamp_uv(uv[best,0], uv[best,1])