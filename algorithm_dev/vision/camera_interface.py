from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.enums import PayloadType
import ctypes
import numpy as np

class Camera:
    def __init__(self):
        # create system and device

        devices = system.create_device()
        if not devices: 
            raise RuntimeError("No camera detected")
        #if len(devices) == 0:
         #   raise RuntimeError("No Lucid cameras detected")

        self.device = devices[0]
        nodemap = self.device.nodemap

        # camera configuration

        nodemap['TriggerMode'].value = 'Off'
        nodemap['AcquisitionMode'].value = 'Continuous'
        nodemap['PixelFormat'].value = 'Mono8'

        # disable chunks entirely 
        nodemap["ChunkModeActive"].value = False

        # --- Stream configuration (TL nodemap, NOT device nodemap) ---
        tl = self.device.tl_stream_nodemap
        tl["StreamBufferHandlingMode"].value = "NewestOnly"
        tl["StreamAutoNegotiatePacketSize"].value = True
        tl["StreamPacketResendEnable"].value = True

        # start streaming
        self.device.start_stream()

        #nodemap["AcquisitionStart"].execute() #try w/o since this may retrigger 

    def get_frame(self, timeout=2000):
        """
        Blocking call that returns a Mono8 numpy array (H, W)
        """ 

        while True:
            buffer = self.device.get_buffer(timeout=timeout)

            try:
                # Only accept image payloads
                #if buffer.payload_type != PayloadType.I:
                 #   continue

                item = BufferFactory.copy(buffer)

                width = item.width
                height = item.height

                # Mono8 → 1 byte per pixel
                num_channels = 1

                # Access raw bytes (Arena View–style)
                c_array = (ctypes.c_ubyte * width * height).from_address(
                    ctypes.addressof(item.pbytes)
                )

                frame = np.ndarray(
                    buffer=c_array,
                    dtype=np.uint8,
                    shape=(height, width)
                )

                # IMPORTANT: copy so downstream code owns memory
                return frame.copy()

            finally:
                self.device.requeue_buffer(buffer)
                if 'item' in locals():
                    BufferFactory.destroy(item)


    def close(self):
        self.device.stop_stream()
        system.destroy_device(self.device)
