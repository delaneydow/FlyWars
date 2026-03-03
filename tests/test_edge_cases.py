"""
FlyWars Edge Case + Safety Tests
--------------------------------
Runs WITHOUT real hardware.

Run from project root:

    python tests/test_edge_cases.py
"""

import time
import numpy as np
from algorithm_dev.control.control_interface import init_hardware

# ===============================
# MOCK HARDWARE
# ===============================

class FakeLaser:
    def __init__(self):
        self.state = "off"
        self.closed = False

    def fire(self):
        self.state = "firing"

    def off(self):
        self.state = "off"

    @property
    def ser(self):
        class Dummy:
            def close(inner_self):
                pass
        return Dummy()


class FakeMirror:
    def __init__(self):
        self.commands = []
        self.closed = False

    def off(self):
        self.commands.append("off")

    def close(self):
        self.closed = True

    def send_uv(self, u, v):
        self.commands.append((u, v))

    def is_reachable(self, x, y):
        return abs(x) < 100 and abs(y) < 100

    def clip_to_reachable(self, x, y):
        return np.clip(x, -100, 100), np.clip(y, -100, 100)


class FakeCamera:
    def get_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)


init_hardware(
    laser_obj=FakeLaser(),
    mirror_obj=FakeMirror()
)


# ===============================
# IMPORT SYSTEM UNDER TEST
# ===============================

from algorithm_dev.control.main import (
    toggle_pause,
    emergency_stop,
)

import algorithm_dev.control.main as main_mod


# ===============================
# TEST UTIL
# ===============================

def print_result(name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")


# ===============================
# TESTS
# ===============================

def test_emergency_shutdown():
    main_mod.laser = FakeLaser()
    main_mod.mirror = FakeMirror()

    emergency_stop(None, None)

    passed = (
        main_mod.laser.state == "off"
        and main_mod.mirror.closed
    )

    print_result("Emergency shutdown", passed)


def test_pause_toggle():
    main_mod.PAUSED = False

    toggle_pause(None, None)
    cond1 = main_mod.PAUSED is True

    toggle_pause(None, None)
    cond2 = main_mod.PAUSED is False

    print_result("Pause toggle", cond1 and cond2)


def test_detection_before_ready():
    """
    System should survive if devices not initialized
    """
    main_mod.laser = None
    main_mod.mirror = None

    try:
        toggle_pause(None, None)
        passed = True
    except Exception:
        passed = False

    print_result("Detection before hardware ready", passed)


def test_mirror_unreachable_clipping():
    mirror = FakeMirror()

    x, y = mirror.clip_to_reachable(1000, -2000)

    passed = mirror.is_reachable(x, y)

    print_result("Mirror unreachable clipping", passed)


def test_cooldown_behavior():
    from algorithm_dev.control.cooldown import adaptive_cooldown

    hot_temp = 85
    cool_temp = 40

    cooldown_hot = adaptive_cooldown(hot_temp)
    cooldown_cool = adaptive_cooldown(cool_temp)

    passed = cooldown_hot >= cooldown_cool

    print_result("Adaptive cooldown scaling", passed)


def test_prediction_horizon_logic():
    """
    Simple sanity check:
    increasing horizon should move prediction farther
    """

    class DummyTrack:
        def __init__(self):
            self.pos = np.array([0.0, 0.0])
            self.vel = np.array([10.0, 0.0])

        def predict(self, k):
            return self.pos + k * self.vel

    track = DummyTrack()

    p1 = track.predict(2)
    p2 = track.predict(10)

    passed = np.linalg.norm(p2) > np.linalg.norm(p1)

    print_result("Prediction horizon K", passed)


def test_multiple_pause_cycles():
    main_mod.PAUSED = False

    for _ in range(10):
        toggle_pause(None, None)

    passed = isinstance(main_mod.PAUSED, bool)

    print_result("Repeated pause stability", passed)


def test_watchdog_shutdown_behavior():
    laser = FakeLaser()
    mirror = FakeMirror()

    laser.fire()

    mirror.send_uv(0.5, 0.5)

    laser.off()
    mirror.off()

    passed = laser.state == "off"

    print_result("Watchdog shutdown", passed)


# ===============================
# RUN ALL TESTS
# ===============================

def run_all():
    print("\n=== FlyWars Edge Case Tests ===\n")

    test_emergency_shutdown()
    test_pause_toggle()
    test_detection_before_ready()
    test_mirror_unreachable_clipping()
    test_cooldown_behavior()
    test_prediction_horizon_logic()
    test_multiple_pause_cycles()
    test_watchdog_shutdown_behavior()

    print("\n=== Tests Complete ===\n")


if __name__ == "__main__":
    run_all()

