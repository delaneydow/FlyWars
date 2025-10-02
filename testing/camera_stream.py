# === FRAME ACQUISITION CLASS DEFINITION ===

""" PURPOSE: this program uses the Arena SDK to test the acquired frames/frame rate 
coming from the LUCID camera for the FlyWars project. The camera will be connected 
via ethernet cable to the radxa x4 board. """

# ===  IMPORTS ===
import cv2
import numpy as np 
from arena_api.system import system

# === DEFINE CLASSES === 
class CameraStream: 
    def __init__(self): 
    # discover connected camera
        devices = system.create_device() 
        if not devices: 
            raise RuntimeError("No camera detected.")
        self.device=devices[0] #TODO; MAY NEED TO MODIFY LATER TO PARSE THROUGH VARIOUS CONNECTED DEVICES (I.E. MIRROR, LASER, ETC.)
    
    def __enter__(self):
        self.stream=self.device.start_stream() 
        return self 
    
    def __exit__(self, exec_type, exec_val, exec_tb): 
        self.stream.__exist__(exec_type, exec_val, exec_tb)

    def get_frame(self):
        buffer = self.device.get_buffer()

        #convert buffer to numpy array
        img = np.copy(np.array(buffer.data, dtype=np.unit8))
        img = img.reshape(buffer.height, buffer.width)) 

        self.device.requeue_buffer(buffer)

        return img


