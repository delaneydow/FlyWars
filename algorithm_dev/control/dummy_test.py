import numpy as np
import pandas as pd

class DummyKF: 
    def __init__(self, x, y, vx, vy, cov_xx, cov_yy): 
        self.statePost = np.array([[x], [y], [vx], [vy]])
        self.errorCovPost = np.diag([cov_xx, cov_yy, 1, 1])


class DummyTrack: 
    def __init__(self, row): 
        self.id = int(row.track_id)
        self.kf = DummyKF(
            row.x, row.y, 
            row.vx, row.vy,
            row.cov_xx, row.cov_yy
        )