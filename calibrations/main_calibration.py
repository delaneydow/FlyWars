# main_calibration.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mirror_camera_mapping import load_calibration, load_all_calibration_images, estimate_spot_area
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def main(): 
    here = Path(__file__).resolve().parent
    json_path = here / "mirror_coordinates.json"

    # load all images
    images,meta = load_all_calibration_images(json_path)

    # estimate spot area distriution
    areas = estimate_spot_area(images)

    print("Area percentiles:", np.percentile(areas, [5, 25, 50, 75, 95]))

    MIN_AREA = np.percentile(areas, 5)
    MAX_AREA = np.percentile(areas, 95)

    print(f"Estimated spot area range: {MIN_AREA:.0f} – {MAX_AREA:.0f} px")

    # 3) now load calibration with area constraints
    mirror_uv, beam_xy, meta = load_calibration(json_path)

       
    print("Loaded samples:", len(mirror_uv))
    print("Mirror UV range:", mirror_uv.min(0), mirror_uv.max(0))
    print("Beam XY range:", beam_xy.min(0), beam_xy.max(0))

    # --- Deduplicate ---
    uv = mirror_uv
    xy = beam_xy
    _, unique_idx = np.unique(uv, axis=0, return_index=True)
    uv = uv[unique_idx]
    xy = xy[unique_idx]
    print(f"Using {len(uv)} unique calibration points")

    # --- Normalize UV ---
    uv_mean = uv.mean(axis=0)
    uv_std = uv.std(axis=0)
    uvn = (uv - uv_mean) / uv_std

    u, v = uvn[:, 0], uvn[:, 1]
    x, y = xy[:, 0], xy[:, 1]

    # --- TPS fit (regularized) ---
    fx = Rbf(u, v, x, function="thin_plate", smooth=1e-2)
    fy = Rbf(u, v, y, function="thin_plate", smooth=1e-2)

    def predict_xy(u_raw, v_raw):
        uvn = (np.column_stack([u_raw, v_raw]) - uv_mean) / uv_std
        return fx(uvn[:,0], uvn[:,1]), fy(uvn[:,0], uvn[:,1])
    

    # --- RMS error ---
    pred = np.column_stack([fx(u,v), fy(u,v)])
    rms = np.sqrt(np.mean(np.sum((pred - beam_xy)**2, axis=1)))
    print("RMS reprojection error:", rms)

    # --- correlation check ---
    corr_u_x = np.corrcoef(uv[:, 0], xy[:,0])[0,1]
    corr_v_y = np.corrcoef(uv[:,1], xy[:,1])[0,1]

    print("corr(u, x):", corr_u_x)
    print("corr(v, y):", corr_v_y)

    # --- monotonicity sanity ---
    order = np.argsort(u)
    du = np.diff(u[order])
    dx = np.diff(x[order])

    order = np.argsort(v)
    dv = np.diff(v[order])
    dy = np.diff(y[order])

    if np.mean(np.sign(du) == np.sign(dx)) < 0.7:
        print("Non-monotonic u → x mapping")

    if np.mean(np.sign(dv) == np.sign(dy)) < 0.7:
        print("Non-monotonic v → y mapping")


    # --- linear sanity model ---
    model_x = LinearRegression().fit(mirror_uv, x)
    model_y = LinearRegression().fit(mirror_uv, y)

    lin_pred = np.column_stack([
        model_x.predict(mirror_uv),
        model_y.predict(mirror_uv)
    ])

    lin_rms = np.mean(np.linalg.norm(beam_xy - lin_pred, axis=1))
    print("Linear model RMS (sanity):", lin_rms)

    # --- Leave-one-out validation ---
    loo_errs = []
    for i in range(len(uv)):
        mask = np.arange(len(uv)) != i
        fx_i = Rbf(u[mask], v[mask], x[mask], function="thin_plate", smooth=1e-2)
        fy_i = Rbf(u[mask], v[mask], y[mask], function="thin_plate", smooth=1e-2)
        xp = fx_i(u[i], v[i])
        yp = fy_i(u[i], v[i])
        loo_errs.append(np.hypot(xp - x[i], yp - y[i]))
    print("LOO RMS error:", np.sqrt(np.mean(np.square(loo_errs))))

    # --- Visual residual sanity ---
    pred_x, pred_y = fx(u, v), fy(u, v)
    plt.figure(figsize=(6,6))
    plt.quiver(x, y, pred_x - x, pred_y - y, angles='xy', scale_units='xy', scale=1, color='r')
    plt.scatter(x, y, c='b', label='Observed')
    plt.title('Calibration Residuals (Red arrows)')
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.legend()
    plt.grid(True)
    plt.show()
    
   
    # --- Bounds safety (for future mirror planning) ---
    u_min, u_max = uv[:,0].min(), uv[:,0].max()
    v_min, v_max = uv[:,1].min(), uv[:,1].max()
    print(f"Mirror command bounds: u [{u_min:.2f}, {u_max:.2f}], v [{v_min:.2f}, {v_max:.2f}]")

    # --- print first few beam coordinates ---
    for (bx, by) in xy[:5]:
        print(f"Beam @ ({bx:.1f}, {by:.1f})")
    
    # generate mirror lookup map 

    GRID = 120 # resolution TODO need to adjust this 

    u_grid = np.linspace(u_min, u_max, GRID)
    v_grid = np.linspace(v_min, v_max, GRID)
    uu, vv = np.meshgrid(u_grid, v_grid)

    uv_flat = np.column_stack([uu.ravel(), vv.ravel()])

    # predict beam positions
    uvn = (uv_flat - uv_mean) / uv_std
    x_map = fx(uvn[:,0], uvn[:,1])
    y_map = fy(uvn[:,0], uvn[:,1])

    mirror_map = {
        "u": uv_flat[:,0],
        "v": uv_flat[:,1],
        "x": x_map,
        "y": y_map, 
        "uv_mean": uv_mean.tolist(), 
        "uv_std": uv_std.tolist(),
        #"bounds": [u_min, u_max, v_min, v_max]
    }

    hull = ConvexHull(xy) # xy = measured beam positions
    hull_pts = xy[hull.vertices]

    np.savez("mirror_map1.npz", 
             **mirror_map, hull=hull_pts)
    print("saved mirror_map1.npz")

    plt.figure(figsize=(6,6))
    plt.scatter(x_map, y_map, s=2)
    plt.title("Mirror reachable beam positions")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
