#control_interface.py
from algorithm_dev.control.planner import plan_targets, LASER_COOLDOWN_FRAMES
import numpy as np
import time
from algorithm_dev.control.laser_interface import LaserInterface
from algorithm_dev.control.mirror_planner import MirrorPlanner
from algorithm_dev.control.object_scoring import SPOT_RADIUS_PX_SAFE, PREDICT_HORIZON, predict_position

mirror_settle_time = 0.025 # 25ms, given rating of settling time + how long to switch directions (avg.) 

# constants
beam_position = np.array([512, 384])  # TODO figure out 0,0 origin, initialize once 

DEBUG_CNTRL = False  # set True only when debugging
DEBUG_SCORE = False 

HIT_VERIFY_INTERVAL = 0.05  # check every 50ms during fire
MIN_HIT_TIME = 0.25          # required hit duration

def fire_with_tracking(laser, mirror, cmd, tracks, track_states):
    """Fire laser while verifying beam stays on target for MIN_HIT_TIME."""

    target_id = cmd["track_id"]
    u, v = mirror.find_uv_for_xy(*cmd["aim"])
    mirror.send_uv(u, v)
    time.sleep(mirror_settle_time)

    # start firing
    laser.ser.write(b"FIRE\n")
    laser.ser.flush()

    #read MCU ack 
    try:
        resp = laser.ser.readline().decode().strip()
    except Exception:
        pass

    fire_start = time.perf_counter()
    hit_time = 0.0
    last_check = fire_start
    settling = False
    settle_start = None

    while hit_time < MIN_HIT_TIME:
        now = time.perf_counter()
        elapsed = now - fire_start

        # safety timeout — don't fire forever
        if now - fire_start > MIN_HIT_TIME * 4:  # hard safety timeout
            print("[FIRE] safety timeout")
            break

          # if mirror is settling after a redirect, wait before counting hit time
        if settling:
            if now - settle_start >= mirror_settle_time:
                settling = False
            else:
                continue

        # check every interval
        if now - last_check >= HIT_VERIFY_INTERVAL:
            last_check = now

            # find the target track
            target = next((t for t in tracks if t.id == target_id), None)
            if target is None:
                print("[FIRE] target lost — ending fire")
                break

            # predict current position
            pred, _ = predict_position(target, k=0)  # k=0 = current state
            dist = float(np.linalg.norm(pred - np.asarray(cmd["aim"])))

            # check if spot covers target — use 2x spot radius as tolerance
            if dist < SPOT_RADIUS_PX_SAFE * 2:
                hit_time += HIT_VERIFY_INTERVAL
            else:
                # only redirect if target has moved more than one spot radius
                if dist > SPOT_RADIUS_PX_SAFE:
                    new_u, new_v = mirror.find_uv_for_xy(*pred)
                    mirror.send_uv(new_u, new_v)
                    cmd["aim"] = pred
                    settling = True
                    settle_start = now
                    print(f"[FIRE] redirecting, dist={dist:.1f}px")

    laser.ser.write(b"OFF\n")
    laser.ser.flush()
    try:
        laser.ser.readline()
    except Exception:
        pass
    laser.ready = True

    total = time.perf_counter() - fire_start
    print(f"[FIRE] hit={hit_time:.3f}s total={total:.3f}s confirmed={hit_time >= MIN_HIT_TIME}")
    return hit_time >= MIN_HIT_TIME



def control_step(tracks, track_states, frame_idx, laser, mirror):
    global beam_position

    if DEBUG_CNTRL:
        print(f"[CONTROL] {len(tracks)} tracks, {len(track_states)} states")

    if not tracks:
        if DEBUG_CNTRL:
            print("[CONTROL] no tracks, skipping")
        return None

    for t in tracks: 

        pred_xy, k_eff = predict_position(t, k=PREDICT_HORIZON + LASER_COOLDOWN_FRAMES)
        t.cached_prediction = pred_xy
        t.cached_k = k_eff
        if DEBUG_SCORE:
            print(f"  [SCORE DEBUG] track {t.id} pred={pred_xy} k_eff={k_eff}")

    # plan targets
    plan = plan_targets(tracks, track_states, beam_position, frame_idx) #TODO i think this is actually mirror, pass where mirror moved to last time it fired? 

    if not plan:
        if DEBUG_CNTRL:
            print("[CONTROL] planner returned empty, skipping")
        return None #do nothing for frame
   

    # fire laser on highest-priority ranked target, first planned shot per frame
    cmd = plan[0] #highest priority target 
    u, v = mirror.find_uv_for_xy(*cmd["aim"]) #compute u & v

    mirror.send_uv(u, v) # use local u, v
    if DEBUG_CNTRL:
        print(f"[FIRE] track={cmd['track_id']} aim={cmd['aim']} uv=({cmd['u']:.3f},{cmd['v']:.3f})")
    #laser.fire()
    hit_confirmed = fire_with_tracking(laser, mirror, cmd, tracks, track_states)
    if DEBUG_CNTRL:
        print(f"[FIRE] laser fired")

    beam_position = cmd["aim"] #stores last position of mirror, updated during tracking

    return {
        "fired": True,
        "frame": frame_idx,
        "track_id": cmd["track_id"],
        "score": cmd["score"],
        "aim_x": float(cmd["aim"][0]),
        "aim_y": float(cmd["aim"][1]),
    }


