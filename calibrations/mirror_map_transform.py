#mirror_map_transform.py

import numpy as np
from scipy.interpolate import Rbf

# ------------------------------------------------------------
# Transform fits
# ------------------------------------------------------------

def fit_translation(old_xy, new_xy):
    delta = new_xy - old_xy
    t = delta.mean(axis=0)
    return {"type": "translation", "t": t}


def fit_similarity(old_xy, new_xy):
    """
    Fit similarity transform: rotation + uniform scale + translation
    """
    old_centroid = old_xy.mean(axis=0)
    new_centroid = new_xy.mean(axis=0)

    X = old_xy - old_centroid
    Y = new_xy - new_centroid

    # SVD-based Procrustes
    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt

    scale = np.trace(np.diag(S)) / np.sum(X**2)

    A = scale * R
    b = new_centroid - A @ old_centroid

    return {"type": "similarity", "A": A, "b": b}


def fit_affine(old_xy, new_xy):
    N = old_xy.shape[0]
    X = np.hstack([old_xy, np.ones((N, 1))])
    Y = new_xy

    T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    A = T[:2, :].T
    b = T[2, :]

    return {"type": "affine", "A": A, "b": b}

def fit_tps(old_xy, new_xy, smooth=1e-3):
    dx = new_xy[:, 0] - old_xy[:, 0]
    dy = new_xy[:, 1] - old_xy[:, 1]

    fx = Rbf(old_xy[:, 0], old_xy[:, 1], dx,
             function="thin_plate", smooth=smooth)
    fy = Rbf(old_xy[:, 0], old_xy[:, 1], dy,
             function="thin_plate", smooth=smooth)

    return {
        "type": "tps",
        "fx": fx,
        "fy": fy
    }


# ------------------------------------------------------------
# Apply transforms
# ------------------------------------------------------------

def apply_transform(model, xy):
    if model["type"] == "translation":
        return xy + model["t"]

    elif model["type"] in ["similarity", "affine"]:
        return (model["A"] @ xy.T).T + model["b"]
    
    elif model["type"] == "tps":
        dx = model["fx"](xy[:, 0], xy[:, 1])
        dy = model["fy"](xy[:, 0], xy[:, 1])
        return xy + np.column_stack([dx, dy])

    else:
        raise ValueError("Unknown transform type")


# ------------------------------------------------------------
# Error metrics
# ------------------------------------------------------------

def compute_error(pred, target):
    errs = np.linalg.norm(pred - target, axis=1)
    return {
        "rmse": np.sqrt(np.mean(errs**2)),
        "mean": np.mean(errs),
        "max": np.max(errs),
    }


# ------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------

def evaluate_models(old_xy, new_xy):
    """models = [
        fit_translation(old_xy, new_xy),
        fit_similarity(old_xy, new_xy),
        fit_affine(old_xy, new_xy),
        fit_tps(old_xy, new_xy),
    ]"""

    model_funcs = [
        ("translation", fit_translation),
        ("similarity", fit_similarity),
        ("affine", fit_affine),
        ("tps", fit_tps),
    ]

    results = []

    """print("\n=== TRANSFORM COMPARISON ===")
    print(f"{'Model':<12} {'RMSE':>8} {'Mean':>8} {'Max':>8}")
    print("-" * 40)

    for m in models:
        pred = apply_transform(m, old_xy)
        err = compute_error(pred, new_xy)

        results.append((m, err))

        print(f"{m['type']:<12} {err['rmse']:8.2f} {err['mean']:8.2f} {err['max']:8.2f}")

    # pick best (lowest RMSE)
    best_model, best_err = min(results, key=lambda x: x[1]["rmse"])

    print("\nBest model:", best_model["type"])
    print(f"RMSE: {best_err['rmse']:.2f}px")

    return best_model, results"""

    print("\n=== TRANSFORM COMPARISON (LOOCV) ===")
    print(f"{'Model':<12} {'RMSE':>8} {'Mean':>8} {'Max':>8}")
    print("-" * 40)

    for name, func in model_funcs:
        err = evaluate_model_loocv(func, old_xy, new_xy)

        model = func(old_xy, new_xy)  # final model on all data
        results.append((model, err))

        print(f"{name:<12} {err['rmse']:8.2f} {err['mean']:8.2f} {err['max']:8.2f}")

    best_model, best_err = min(results, key=lambda x: x[1]["rmse"])

    print("\nBest model:", best_model["type"])
    print(f"RMSE: {best_err['rmse']:.2f}px")

    return best_model, results


def apply_transform_to_map(map_path_in, map_path_out, model):
    data = np.load(map_path_in)

    xy = np.column_stack([data["x_map"], data["y_map"]])

    xy_new = apply_transform(model, xy)

    np.savez(
        map_path_out,
        uv=data["uv"],
        xy=data["xy"],
        uv_mean=data["uv_mean"],
        uv_std=data["uv_std"],
        x_map=xy_new[:, 0].astype(np.float32),
        y_map=xy_new[:, 1].astype(np.float32),
        u_map=data["u_map"],
        v_map=data["v_map"],
        bounds=data["bounds"],
        hull=data["hull"],
        hull_raw=data["hull_raw"],
        spot_radius_px=data["spot_radius_px"]
    )

    print(f"\n✅ Saved corrected map → {map_path_out}")
    print("Original map unchanged.")


def evaluate_model_loocv(fit_func, old_xy, new_xy):
    """
    Leave-one-out cross validation
    """
    preds = []

    for i in range(len(old_xy)):
        mask = np.ones(len(old_xy), dtype=bool)
        mask[i] = False

        model = fit_func(old_xy[mask], new_xy[mask])
        pred = apply_transform(model, old_xy[i:i+1])[0]

        preds.append(pred)

    preds = np.array(preds)
    err = compute_error(preds, new_xy)

    return err



"""
# STEP 1: Load your 9-point data
old_xy = np.array([...])  # from original calibration
new_xy = np.array([...])  # from field measurement

# STEP 2: Evaluate models
best_model, results = evaluate_models(old_xy, new_xy)

# OPTIONAL: force a specific model instead
# best_model = [r[0] for r in results if r[0]["type"] == "affine"][0]

# STEP 3: Apply to map
apply_transform_to_map(
    "mirror_map.npz",
    "mirror_map_corrected.npz",
    best_model
)

"""

""" arrays follow format: 
(0.1, -0.1), (0.1, 0), (0.1, 0.1),
(0.0, -0.1), (0.0, 0), (0.0, 0.1),
(-0.1, -0.1), (-0.1, 0), (-0.1, 0.1)
"""
old_xy = np.array([(637.5, 993.8), (628.2, 675.3), (648.3, 361.9),
                    (1075.4, 979.3), (1069.2, 673.9), (1083.2, 372.9),
                    (1514.4, 966.2), (1505.3, 678.4), (1513.1, 395.1)]) # original calibration
new_xy = np.array([(551, 1008), (503, 716), (502, 408),
                    (1013, 972), (1006, 666), (1020, 364),
                    (1446, 960), (1440, 670), (1452, 386)]) # from field measurement

best_model, results = evaluate_models(old_xy, new_xy)




