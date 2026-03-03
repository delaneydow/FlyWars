class Writer:
    def __init__(self): 

        self.frame_file = open("frames.csv", "w", buffering=1)
        self.track_file = open("tracks.csv", "w", buffering=1)
        self.fire_file = open("fires.csv", "w", buffering=1)


        self.frame_file.write("frame,time,latency_ms,pipeline_latency_ms,ndet,ntrack,temp,cooldown\n")

        self.track_file.write("frame,track,x,y,vx,vy,speed,state,cov\n")

        self.fire_file.write("frame,track,score,x,y\n")

    def log_frame(self, f):
            self.frame_file.write(
                f"{f['frame']},{f['time']},{f['vision_latency_ms']}, {f['pipeline_latency_ms']},"
                f"{f['ndet']},{f['ntrack']},"
                f"{f['temp']},{f['cooldown']}\n"
            )

    


