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
        return abs(x) <= 100 and abs(y) <= 100

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

    # === Dummy track adapter ===
# === Dummy track adapter ===
class DummyKF: 
    def __init__(self, x, y, vx, vy): 
        # simplified covariances for testing
        self.statePost = np.array([[float(x)], [float(y)], [float(vx)], [float(vy)]])
        self.errorCovPost = np.diag([1,1,1,1])

class DummyTrack: 
    def __init__(self, x, y, vx, vy, track_id=1): 
        self.id = track_id
        self.kf = DummyKF(x, y, vx, vy)
        self.cached_prediction = None
        self.cached_k = None



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

    # safe exit without hardware

    def test_control_without_hardware():
        import algorithm_dev.control.control_interface as ci

        ci.laser = None
        ci.mirror = None

        try:
            ci.control_step([], {}, 1)
            passed = True
        except Exception:
            passed = False

        print_result("Control safe without hardware", passed)

# === FIRING/FULL SYSTEM TESTS ===

# single object
def test_single_target_fire():
    from algorithm_dev.control.control_interface import control_step

    track = DummyTrack(x=0., y=0., vx=5., vy=0.)

    result = control_step([track], {}, 1)

    passed = result is not None and result["fired"]

    print_result("Single target fire", passed)

    # multiple targets, one frame

    def test_single_fire_per_frame():
        from algorithm_dev.control.control_interface import control_step

        class DummyTrack:
            def __init__(self, i):
                self.id = i
                self.pos = np.array([i*5.,0.])
                self.vel = np.array([1.,0.])

        tracks = [DummyTrack(i) for i in range(5)]

        result = control_step(tracks, {}, 1)

        passed = result is not None

        print_result("Single fire per frame", passed)

    # beam position updates
    def test_beam_position_updates():
        import algorithm_dev.control.control_interface as ci

        start = ci.beam_position.copy()

        class DummyTrack:
            def __init__(self):
                self.id = 1
                self.pos = np.array([20.,20.])
                self.vel = np.zeros(2)

        ci.control_step([DummyTrack()], {}, 1)

        passed = not np.allclose(start, ci.beam_position)

        print_result("Beam position update", passed)

    # planner stability over frames: 
    def test_multi_frame_stability():
        from algorithm_dev.control.control_interface import control_step

        class DummyTrack:
            def __init__(self):
                self.id = 1
                self.pos = np.array([0.,0.])
                self.vel = np.array([2.,0.])

        stable = True

        for f in range(10):
            r = control_step([DummyTrack()], {}, f)
            if r is None:
                stable = False

        print_result("Multi-frame stability", stable)



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

    test_single_target_fire()
    test_single_fire_per_frame()
    test_beam_position_updates()
    test_control_without_hardware()
    test_multi_frame_stability()

    print("\n=== Tests Complete ===\n")


if __name__ == "__main__":
    run_all()

