#analyze_video.py

import cv2
import numpy as np

# at 25fps, each frame = 40ms
# 0.25s = 6.25 frames exactly
# so 6 frames = 240ms (just under), 7 frames = 280ms (confirmed)
# we need to account for this sampling uncertainty

def analyze_video(video_path, min_hit_time = 0.25, spot_threshold=220, min_spot_area=5): 

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps

    #sampling unvertainty --> +/- one frame duration at each edge
    sampling_uncertainty = frame_duration

    # === UNCERTAINTY REPORT ===
    print(f"Video: {fps:.1f}fps, {frame_duration*1000:.1f}ms/frame")
    print(f"Sampling uncertainty: ±{sampling_uncertainty*1000:.1f}ms per edge")
    print(f"Min hit threshold: {min_hit_time*1000:.0f}ms")
    print(f"Frames needed for confirmed hit: {int(np.ceil(min_hit_time / frame_duration))} frames")
    print(f"Frames needed for conservative confirmed hit: "
          f"{int(np.ceil((min_hit_time + 2*sampling_uncertainty) / frame_duration))} frames")
    print()

    hit_frames = []
    frame_idx = 0

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

        # mono8 - frame already single channel 

        # === EXTRACT CENTROIDS === 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        _, spot_mask = cv2.threshold(gray, spot_threshold, 255, cv2.THRESH_BINARY)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(spot_mask)

        spot_found = False
        best_area = 0
        spot_x, spot_y, spot_area = 0, 0, 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_spot_area and area > best_area:
                best_area = area
                spot_x = centroids[i][0]
                spot_y = centroids[i][1]
                spot_area = area
                spot_found = True
                break #take largest spot 

        if spot_found:
            hit_frames.append({
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "spot_x": spot_x,
                "spot_y": spot_y,
                "spot_area": spot_area,
            })
        
        frame_idx += 1
    cap.release()
    total_frames = frame_idx

    # find consecutive hit sequences to measure time 
    if not hit_frames:
        print("No laser spot detected in video")
        return []
    
    # === GROUP INTO CONTINUOUS HITS ===
    sequences = []
    seq_frames = [hit_frames[0]]

    for i in range(1, len(hit_frames)):
        gap = hit_frames[i]["frame"] - hit_frames[i-1]["frame"]
        if gap <= 3:
            seq_frames.append(hit_frames[i])
        else:
            sequences.append(seq_frames)
            seq_frames = [hit_frames[i]]
    sequences.append(seq_frames)

    # analyze each sequence with confidence intervals
    results = []
    for seq in sequences:
        n_frames = len(seq)
        measured_duration = n_frames * frame_duration

        # confidence interval accounting for sampling at both edges
        # worst case: laser started just after first captured frame,
        # ended just before last captured frame
        duration_min = max(0, (n_frames - 1) * frame_duration)
        # best case: laser started just before first frame, ended just after last
        duration_max = (n_frames + 1) * frame_duration

        # confirmed if even worst-case estimate meets threshold
        conservative_confirmed = duration_min >= min_hit_time
        # likely confirmed if measured duration meets threshold
        likely_confirmed = measured_duration >= min_hit_time
        # possible if best-case meets threshold
        possible = duration_max >= min_hit_time

        result = {
            "start_frame": seq[0]["frame"],
            "end_frame": seq[-1]["frame"],
            "n_frames": n_frames,
            "measured_duration_ms": measured_duration * 1000,
            "duration_min_ms": duration_min * 1000,
            "duration_max_ms": duration_max * 1000,
            "conservative_confirmed": conservative_confirmed,
            "likely_confirmed": likely_confirmed,
            "possible": possible,
            "mean_spot_x": np.mean([f["spot_x"] for f in seq]),
            "mean_spot_y": np.mean([f["spot_y"] for f in seq]),
            "mean_spot_area": np.mean([f["spot_area"] for f in seq]),
        }
        results.append(result)

    # summary
    conservative = [r for r in results if r["conservative_confirmed"]]
    likely       = [r for r in results if r["likely_confirmed"]]
    possible     = [r for r in results if r["possible"]]

    print(f"Total sequences detected: {len(results)}")
    print(f"Conservative confirmed (>=250ms even worst-case): {len(conservative)}")
    print(f"Likely confirmed (measured >=250ms):               {len(likely)}")
    print(f"Possible (>=250ms best-case only):                 {len(possible)}")
    print()

    for i, r in enumerate(results):
        status = "CONFIRMED" if r["conservative_confirmed"] else \
                 "LIKELY   " if r["likely_confirmed"] else \
                 "POSSIBLE " if r["possible"] else "MISS     "
        print(
            f"  [{status}] frames {r['start_frame']:4d}-{r['end_frame']:4d} "
            f"| {r['n_frames']} frames "
            f"| {r['measured_duration_ms']:.0f}ms measured "
            f"| [{r['duration_min_ms']:.0f}-{r['duration_max_ms']:.0f}ms] CI "
            f"| spot=({r['mean_spot_x']:.0f},{r['mean_spot_y']:.0f})"
        )

    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "hit_test.avi"
    min_hit = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 220
    analyze_video(path, min_hit_time=min_hit, spot_threshold=threshold)
