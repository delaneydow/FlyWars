#control_interface.py
#control_interface.py
from algorithm_dev.control.planner import plan_targets, LASER_COOLDOWN_FRAMES
import numpy as np
import time
from algorithm_dev.control.object_scoring import SPOT_RADIUS_PX_SAFE, PREDICT_HORIZON, predict_position 

# constants
beam_position = np.array([989.6, 620.2])  # initialized once, taken from 0,0 origin of mirror/beam centroid 

DEBUG_CNTRL = False  # set True only when debugging
DEBUG_SCORE = False

HIT_VERIFY_INTERVAL = 0.010  # check every 10ms during fire
MIN_HIT_TIME_MS = 250          # required hit duration (ms)

def fire_with_tracking(laser, mirror, cmd, tracks, track_states):
    """Fire laser while verifying beam stays on target for MIN_HIT_TIME."""

    target_id = cmd["track_id"]
    u, v = mirror.find_uv_for_xy(*cmd["aim"])
    mirror.send_uv(u, v) # blocks until settled - safe to fire immediately after

    # start firing
    laser.ser.write(f"FIRE {MIN_HIT_TIME_MS}\n".encode()) #send timed fire command - MCU owns time keeping for 250 ms
    laser.ser.flush()

    #read MCU ack 
    try:
        laser.ser.readline().decode().strip()
    except Exception:
        pass

    fire_start = time.perf_counter()
    #hit_time = 0.0
    redirect_count = 0
    aborted = False 


    # loop redirects laser or aborts -- duration is job of MCU
    while time.perf_counter() - fire_start < (MIN_HIT_TIME_MS / 1000) * 4:
        now = time.perf_counter()
        elapsed = now - fire_start

        # check for MCU done signal (non-blocking)
        if laser.ser.in_waiting: 
            msg = laser.ser.readline().decode().strip()
            if msg == "DONE":
                break # MCU finished its time window

        if elapsed < (MIN_HIT_TIME_MS / 1000): #still within expected window
            target = next((t for t in tracks if t.id == target_id), None)
            if target is None: 
                laser.ser.write(b"OFF\n")
                laser.ser.flush()
                aborted = True
                break
            
            pred, _ = predict_position(target, k=0) # k=0, current state
            dist = float(np.linalg.norm(pred - np.asarray(cmd["aim"])))

        
            if dist > SPOT_RADIUS_PX_SAFE:
                new_u, new_v = mirror.find_uv_for_xy(*pred)
                mirror.send_uv(new_u, new_v)
                cmd["aim"] = pred
                settling = True
                settle_start = now
                redirect_count += 1
                #print(f"[FIRE] redirecting, dist={dist:.1f}px")

        time.sleep(HIT_VERIFY_INTERVAL)

    laser.ready = True
    total = time.perf_counter() - fire_start

    #print(f"[FIRE] hit={hit_time:.3f}s total={total:.3f}s confirmed={hit_time >= MIN_HIT_TIME}")
    return {
        "confirmed": not aborted,
        "hit_time": round(min(total, MIN_HIT_TIME_MS / 1000), 3),
        "total_time": round(total, 3),
        "redirects": redirect_count,
    }


def control_step(tracks, track_states, frame_idx, laser, mirror, suppression=None):
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
    #u, v = mirror.find_uv_for_xy(*cmd["aim"]) #compute u & v
    #mirror.send_uv(u, v) # use local u, v
    if DEBUG_CNTRL:
        print(f"[FIRE] track={cmd['track_id']} aim={cmd['aim']}")
    #laser.fire()
    t_start = time.perf_counter()
    hit_result = fire_with_tracking(laser, mirror, cmd, tracks, track_states)
    t_end = time.perf_counter()
    # === TRIGGER SUPPRESSION IMMEDIATELY AFTER FIRE ===
    if suppression is not None:
        # supppress_frames covers laser flash duration + mirror setttling 
        fire_frames = int((MIN_HIT_TIME_MS / 1000) * 120) + 4 # +4 frames settling
        suppression.trigger(frames=fire_frames)
    if DEBUG_CNTRL:
        print(f"[FIRE] laser fired for {(t_end-t_start)*1000:.1f}ms")

    beam_position = cmd["aim"] #stores last position of mirror, updated during tracking

    return {
        "fired": True,
        "frame": frame_idx,
        "track_id": cmd["track_id"],
        "score": cmd["score"],
        "aim_x": float(cmd["aim"][0]),
        "aim_y": float(cmd["aim"][1]),
        "hit_time": hit_result["hit_time"],
        "hit_confirmed": hit_result["confirmed"],
        "redirects": hit_result["redirects"]
    }
