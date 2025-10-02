""" PURPOSE: 
This tests various object detection models from the streaming camera 
and compares performance

TODO; ADD ADDITIONAL MODELS
TODO; ADD ADDITIONAL PERFORMANCE STATS (IF NEEDED)
TODO; CHECK RGB VS GRAYSCALE PERFORMANCE
TODO; CHECK RUNTIME OF WRITING TO FILES"""

import cv2
import time
import csv
from camera_stream import CameraStream

# choose detection models
from ultralytics import YOLO  # YOLOv8
# from some_other_model import otherModel  # TODO; pull different models to compare 

# Initialize models
models = {
    "yolov8n": YOLO("yolov8n.pt"),
    # "custom_model": YourOtherModel("weights.pth"),
}

OUTPUT_CSV = "detection_stats.csv"
def main():

    # Prepare CSV file
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame_num", "model_name", "inference_time", "num_detections"]
        writer.writerow(header)

        frame_num = 0

    with CameraStream() as cam:
        while True:
            frame_num += 1
            frame = cam.get_frame()

            # Convert grayscale → RGB for models that expect 3 channels, TODO; see how grayscale vs. RGB impacts performance (may depend on experiment settings)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            for name, model in models.items():
                    start = time.time()
                    results = model(frame_rgb)
                    end = time.time()
                    inference_time = end - start
                    num_detections = len(results[0].boxes)

                    # Write stats to CSV
                    writer.writerow([frame_num, name, f"{inference_time:.4f}", num_detections]) #TODO; double check that "writing to file" doesn't significantly impede performance, should be O(n) runtime 

                    # Optional: display annotated frame
                    annotated = results[0].plot()
                    cv2.imshow(f"{name} Detection", annotated)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

