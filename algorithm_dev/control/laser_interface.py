# laser_interface.py

class LaserInterface:
    def __init__(self):
        self.ready = True

    def send(self, x, y):
        """
        Send coordinates to laser hardware
        """
        # TODO: replace with SDK / serial / GPIO
        print(f"[LASER] firing at ({x:.1f}, {y:.1f})")

    def fire(self, aim_point):
        if self.ready:
            self.send(*aim_point)
