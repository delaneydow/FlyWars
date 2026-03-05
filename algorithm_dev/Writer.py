import numpy as np

class Writer:
    def __init__(self): 

        self.frame_file = open("frames.csv", "w", buffering=1)
        self.track_file = open("tracks.csv", "w", buffering=1)
        self.fire_file = open("fires.csv", "w", buffering=1)


        self.frame_file.write("frame,time,vision_latency_ms,pipeline_latency_ms,ndet,ntrack,temp,cooldown\n")

        self.track_file.write("frame,track,x,y,vx,vy,speed,state,cov\n")

        self.fire_file.write("frame,track,score,x,y\n")

    def log_frame(self, f):
            self.frame_file.write(
                f"{f['frame']},{f['time']},{f['vision_latency_ms']}, {f['pipeline_latency_ms']},"
                f"{f['ndet']},{f['ntrack']},"
                f"{f['temp']},{f['cooldown']}\n"
            )
    def log_fire(self, result):
        if not result:
            return
        self.fire_file.write(
                f"{result['frame']},{result['track_id']},{result['score']:.4f},"
                f"{result['aim_x']:.1f},{result['aim_y']:.1f},"
                f"{result.get('hit_time', 0.0):.3f},"
                f"{result.get('hit_confirmed', False)},"
                f"{result.get('redirects', 0)}\n"
            )
        
    def log_track(self, frame_idx, tracks, track_states): 
        """Call this each frame to record track states."""
        for t in tracks:
            vx = float(t.kf.statePost[2, 0])
            vy = float(t.kf.statePost[3, 0])
            x  = float(t.kf.statePost[0, 0])
            y  = float(t.kf.statePost[1, 0])
            speed = float(np.hypot(vx, vy))
            cov = float(t.kf.errorCovPost[0,0] + t.kf.errorCovPost[1,1])
            state = track_states.get(t.id, -1)
            self.track_file.write(
                f"{frame_idx},{t.id},{x:.1f},{y:.1f},"
                f"{vx:.2f},{vy:.2f},{speed:.2f},{state},{cov:.2f}\n"
            ) 

    def close(self):
        self.frame_file.close()
        self.track_file.close()
        self.fire_file.close() 
    


