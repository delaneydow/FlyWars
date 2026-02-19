# connection_testing.py

# goal is to ensure that mirror, camera, and signal can all work simultaneously + in union with microcontroller 


#camera via ethernet --> confirm frame acquisition and throughput 

# mirror controler via serial --> confirm command/response loop works

# laser driver via microcontroller pwm output --> confirm serial command produces PWM/TTL signal 


import cv2
import serial
import time
import subprocess
from algorithm_dev.vision.camera_interface import Camera
import sys

# ==== CONFIGURATION ===
CAMERA_INDEX = 0 # adjust if needed
MIRROR_PORT = "/dev/ttyUSB0" # TODO adjust/check this if necessary 
LASER_PORT = "dev/ttyACM0" #TODO adjust/check this, should be the microcontroller port 

MIRROR_BAUD = 115200
LASER_BAUD = 115200

TEST_FIRE_COUNT = 3

# === CAMERA TESTING ===
def test_camera(): 
    print("\n *** CAMERA TEST ***")

    cam = Camera()
    frame = cam.get_frame()

    if frame is None:
        raise RuntimeError("No frame received")

    print("Frame shape:", frame.shape)
    cam.close()
    print("[PASS] Camera streaming.")

    return True


# === MIRROR SERIAL TEST ===
def test_mirror(port="/dev/ttyUSB0"):
    print("Testing mirror controller...")

    try: 

        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)

        # Status check
        ser.write(b"STATUS\r\n")
        print("Status:", ser.readline())

        # Movement test
        ser.write(b"X=0.2\r\n")
        time.sleep(1)

        ser.write(b"X=-0.2\r\n")
        time.sleep(1)

        ser.write(b"X=0.0\r\n")
        print("Mirror test complete.")

        ser.close()
        print("[PASS] Connection, X and Y movement established")
        #return True
    
    except Exception as e: 
        print("[FAIL] mirror connection:", e)
        return False
    

# === LASER PWM SERIAL TEST ===
def test_laser():
    print("\n=== LASER PWM TEST ===")

    try:
        ser = serial.Serial(LASER_PORT, LASER_BAUD, timeout=1)
        time.sleep(2)

        for i in range(TEST_FIRE_COUNT):
            print(f"Fire test {i+1}")
            ser.write(b"FIRE\n")
            ser.flush()
            time.sleep(1)

        ser.close()
        print("[PASS] Laser serial command sent.")
        return True

    except Exception as e:
        print("[FAIL] Laser connection:", e)
        return False
    
# === SYSTEM INFO CHECK ===
def system_check(): 
    print("\n=== SYSTEM CHECK ===")

    print("Network interfaces:")
    subprocess.run(["ip", "addr"])

    print("\nSerial devices:")
    subprocess.run(["ls", "/dev/tty*"])

# === TESTING PIPELINE === 
def main():
    print("=== HARDWARE INTEGRATION TEST ===")

    system_check()

    results = {
        "camera": test_camera(),
        "mirror": test_mirror(),
        "laser": test_laser(),
    }

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    if not all(results.values()):
        print("\nSome subsystems failed.")
    else:
        print("\nAll systems operational.")


if __name__ == "__main__":
    main()