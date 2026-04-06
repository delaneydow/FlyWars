#fire_suppression.py 

"""Communication between control and vision pipelines. This attempts to mitigate the effects
after the laser fires, since the beam flash can lead to false detections and track explosion.
This approach uses a shared counter between tracker and frame differencing to block new tracks + 
drop detections during the time of laser firing  """

import threading

class FireSuppression:
    """Shared counter between control and vision layer.
    Thread-safe. Pass a single instance to process_video and control_step"""

    def __init__(self, suppress_frames: int = 8):
        self._lock = threading.Lock()
        self._remaining = 0
        self.suppress_frames = suppress_frames #might need to tune for flash duration

    
    def trigger(self, frames: int = None): 
        """ Call immediately after firing laser """
        with self._lock:
            self._remaining = frames if frames is not None else self.suppress_frames

    def tick(self) -> bool: 
        """Call once per vision frame. Returns True if currently suppressed. Decrements counter automatically."""
        with self._lock:
            if self._remaining > 0: 
                self._remaining -=1
                return True
            return False
    @property
    def active(self) -> bool: 
        with self._lock: 
            return self._remaining > 0 