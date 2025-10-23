import time
import numpy as np 
import cv2
from scipy.linalg import lstsq
from sklearn.model_selection import train_test_split

# === SDK PLACEHOLDERS ===
class MirrorSDK: 
    def __init__(self): pass 


class CameraSDK: 
    def __init__(self): pass 


# instantiate classes 
mirror = MirrorSDK()
camera = CameraSDK() 

# === CREATE HELPER FUNCTIONS ===

#image helpers
def find_laser_spot(frame, threshold=220): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # adaptive, treshold near max value
    _,bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    M = cv2.moments(bw) 
    if M["m00"] == 0 :  #TODO; figure out what this is doing 
        #fallback, max intensity
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        return maxLoc # (x, y) 
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy) 

#fit helpers (2nd degree polynomial)
def design_matrix_xy(x,y): 
    # terms: [1, x, y, x*y, x^2, y^2]
    return np.vstack([np.ones_like(x), x, y, x*y, x*x, y*y]).T

def fit_poly2(xy, uv): 
    x = xy[:, 0]; y = xy[:, 1]
    A = design_matrix_xy(x,y)
    coeffs, _, _, _ = lstsq(A, uv) 
    return coeffs # shape (6,2) if uv is (N,2)

def eval_poly2(coeffs, x, y): #test fit
    Arow = np.array([1, x, y, x*y, x*x, y*y])
    out = Arow.dot(coeffs)
    return out # (u, v)

# === DATA COLLECTION ===
def collect_samples(x_positions, y_positions, show_live=True, n_avg=3): 

    samples = []

    if show_live: 
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)


    print ("Starting calibration ...")

    for x in x_positions: 
        for y in y_positions: 
            mirror.set_position(x,y)
            # optional: wait for mirror stability, depends on hardware integration
            t0 = time.time()
            while not mirror.is_stable() and (time.time() - t0) < 2.0: 
                time.sleep(0.1)

            # average detected laser points
            detected_points = []
            for _ in range(n_avg): 
                frame = camera.get_frame()
                spot = find_laser_spot(frame)
                if spot: 
                    detected_points.append(spot)

                # visualization
                if show_live: 
                    vis = frame.copy()
                    if spot: 
                        cv2.circle(vis, (int(spot[0]), int(spot[1])), 6, (0, 0, 255), -1)
                    cv2.putText(vis, f"Mirror ({x:.2f}, {y:.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Calibration", vis)
                    if cv2.waitKey(10) == 27:  # ESC to quit
                        cv2.destroyAllWindows()
                        return np.array(samples)

                time.sleep(0.5)

        if detected_points: 
            mean_spot = np.mean(detected_points, axis=0)
            samples.append((x, y, mean_spot[0], mean_spot[1]))

    if show_live: 
        cv2.destroyAllWindows()
    
    samples = np.array(samples)
    print(f"Collected {len(samples)} valid samples.")
    return samples 

# === CALIBRATION ===

def run_calibration(): 
    # define a grid of mirror positions in normalized range
    x_positions = np.linspace(-0.8, 0.8, 7)
    y_positions = np.linspace(-0.8, 0.8, 7)

    print("Starting mirror sweep and image capture...")
    data = collect_samples(x_positions, y_positions, show_live=True)

    if data.shape[0] < 20: 
        print("Warning: very few samples collected.")


    # split samples into training and validation sets 
    train_data, val_data = train_test_split(data, test_size = 0.2, random_state=42)

    xy_train = train_data[:, 0:2]
    uv_train = train_data[:, 2:4]

    # fit polynomial (mirror --> camera)
    mirror_to_camera = fit_polynomial(xy_train, uv_train)

    # evaluate performance on validation set
    xy_val = val_data[:, 0:2]
    uv_val = val_data[:, 2:4]

    predictions = np.array([eval_polynomial(mirror_to_camera, x, y) for x, y in xy_val])
    rms_error = np.sqrt(np.mean((predictions - uv_val) ** 2))
    print(f"Validation rms pixel error: {rms_error: .3f}px")

    # try fitting inverse mapping for targeting
    U, V = train_data[:, 2], train_data[:, 3]
    A = np.vstack([np.ones_like(U), U, V, U*V, U*U, V*V]).T
    coeff_inv_x, _, _, _ = lstsq(A, train_data[:, 0])
    coeff_inv_y, _, _, _ = lstsq(A, train_data[:, 1])
    camera_to_mirror = np.vstack([coeff_inv_x, coeff_inv_y]).T

    print("Calibration complete.")
    return {
        "mirror to camera": mirror_to_camera,
        "camera to mirror:" camera_to_mirror
    }

# === MAIN FUNCTION ===
if __name__ == "__main__": 
    models=run_calibration()



    

