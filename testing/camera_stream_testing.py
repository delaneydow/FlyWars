""" PURPOSE: 
Pulls from CameraStream class to generate a log/basic performance information 
about the camera before deploying any detection models """

import cv2
import time
import csv 
from camera_stream import CameraStream


OUTPUT_CSV = "camera_fps_log.csv"

def main():
    with CameraStream() as cam:
        frame_count = 0
        start_time = time.time()
        log_data = []

        while True:
            frame = cam.get_frame()
            frame_count += 1

            # display frame for visual check
            cv2.imshow("Camera Stream", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

            # Log FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                print(f"FPS: {fps:.2f}")
                log_data.append([frame_count, fps])
                start_time = time.time()

        # Save log to CSV
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_count", "fps"])
            writer.writerows(log_data)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
