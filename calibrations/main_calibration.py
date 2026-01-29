# main_calibration.py

import cv2
import numpy as np
from mirror_camera_mapping import load_calibration, load_all_calibration_images, estimate_spot_area
from pathlib import Path

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
    mirror_uv, beam_xy, meta = load_calibration(
        json_path,
        min_area=MIN_AREA,
        max_area=MAX_AREA
    )
    

    print("Loaded samples:", len(mirror_uv))
    print("Mirror UV range:", mirror_uv.min(0), mirror_uv.max(0))
    print("Beam XY range:", beam_xy.min(0), beam_xy.max(0))

    # sanity check - monotonicity (physics sanity)
    # mirror motion --> beam motion must be monotonic

    uv = np.asarray(mirror_uv)
    xy = np.asarray(beam_xy)

    du = np.diff(uv[:,0])
    dx = np.diff(xy[:,0])
    dv = np.diff(uv[:,1])
    dy = np.diff(xy[:,1])

    if np.mean(np.sign(du) == np.sign(dx)) < 0.7: 
        print ("Non-monotonic u --> x mapping")
    if np.mean(np.sign(dv) == np.sign(dy)) < 0.7: 
        print ("Non-monotonic v --> y mapping")

    corr_u_x = np.corrcoef(uv[:, 0], xy[:,0])[0,1]
    corr_v_y = np.corrcoef(uv[:,1], xy[:,1])[0,1]

    print("corr(u, x):", corr_u_x)
    print("corr(v, y):", corr_v_y)


    # sanity plots
    for (x,y) in beam_xy[:5]: 
        print(f"Beam @ ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()
