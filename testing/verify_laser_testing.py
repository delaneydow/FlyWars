# verify_laser_testing.py

#vision based hit confirmation

import numpy as np

def verify_laser_on_target(frame, aim_xy, spot_radius_px):
    """Check if bright laser spot appears near aim point in frame."""
    x, y = int(aim_xy[0]), int(aim_xy[1])
    r = int(spot_radius_px * 3)  # search radius

    # crop ROI around aim point
    h, w = frame.shape[:2]
    x0 = max(0, x - r)
    x1 = min(w, x + r)
    y0 = max(0, y - r)
    y1 = min(h, y + r)

    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return False, 0.0

    # laser spot will be near-saturated (>200 for Mono8)
    bright_pixels = np.sum(roi > 200)
    bright_ratio = bright_pixels / roi.size

    return bright_ratio > 0.05, bright_ratio  # 5% of ROI is bright