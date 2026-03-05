from collections import deque
import cv2
import numpy as np

# constants
HISTORY = 50 # frame to keep for plotting


# === TRACKING CLASS ===
# used in tracking.py
class Track: 
    def __init__(self, track_id, centroid): 
        self.id = track_id
        self.centroids = deque(maxlen=HISTORY)
        self.centroids.append(centroid)
        self.missed = 0 # num of consecutive frames w/o detection
        #self.last_seen = 1 # num. of frames since last scene aka how old 
        self.last_seen = 0
        self.alpha = 0.3

        # prediction cache
        self.cached_prediction = None
        self.cached_k = None

        # ADD KALMAN STATE
        # using initial vector [x, y, vx, vy]T
        self.kf = cv2.KalmanFilter(4,2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]], dtype=np.float32)

        self.kf.measurementMatrix=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)
        
        self.q = 1e-2
        
        # tune noise matrices to reduce overshoot 
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 5 #initialize velocity covariance
        self.kf.processNoiseCov[:] = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0 #1e-0 TODO tune this 

        self.kf.statePre = np.array([[centroid[0]],
                                     [centroid[1]],
                                     [0],
                                     [0]], dtype=np.float32)

        self.kf.statePost = self.kf.statePre.copy()

    def predict(self, dt): 
        self.kf.transitionMatrix[0,2] = dt
        self.kf.transitionMatrix[1,3] = dt

        state = getattr(self, "state", "hovering")

        #dyanamic q
        if state == "hovering": 
            q = self.q * 0.2
        elif state == "accelerating": 
            q = self.q * 3.0
        else: 
            q = self.q

        dt2 = dt * dt
        # rebuild process noise covariance
        self.kf.processNoiseCov[:] = 0 
        self.kf.processNoiseCov[0,0] = q * dt2
        self.kf.processNoiseCov[1,1] = q * dt2
        self.kf.processNoiseCov[2,2] = q
        self.kf.processNoiseCov[3,3] = q 

        pred = self.kf.predict()
        # cap covariance to prevent runaway divergence during missed frames
        np.clip(self.kf.errorCovPost, 0, 500, out=self.kf.errorCovPost)
        return pred[0,0], pred[1,0]

    def update(self, detection=None, dt=1/60.0): #TODO use effective dt = frame_dt + system_latency
        # prediction already happens with predicted = {t: t.predict()} in association / tracking helper
        #self.kf.predict() #ensure filter doesn't get stuck correcting a static state
        if detection is not None: 
            # centroid stabilization
            if len(self.centroids) > 0:
                px, py = self.centroids[-1]
                alpha = 0.6

                detection = (
                    alpha*detection[0] + (1-alpha)*px,
                    alpha*detection[1] + (1-alpha)*py
                )
            measured = np.array([[np.float32(detection[0])],
                                 [np.float32(detection[1])]])

            # bootstrap on first detection for velocity bootstrap timing to minimize "hovering lock"
            if len(self.centroids) >=1:
                dx = detection[0] - self.centroids[-1][0]
                dy = detection[1] - self.centroids[-1][1]
                #dx = self.centroids[-1][0] - self.centroids[-2][0]
                #dy = self.centroids[-1][1] - self.centroids[-2][1]

                dt_eff = 2 * dt

                # smoothed velocity --- utilize alpha value 
                self.kf.statePost[2,0] = (1-self.alpha) *self.kf.statePost[2,0] + self.alpha*(dx/ dt_eff)
                self.kf.statePost[3,0] = (1-self.alpha) *self.kf.statePost[3,0] + self.alpha*(dy /dt_eff)

            self.kf.correct(measured)
            MAX_SPEED_PX = 600 #pixels/second, approx 1/2 the FOV per frame 
            vx = self.kf.statePost[2,0]
            vy = self.kf.statePost[3,0]
            speed = np.hypot(vx,vy)
            if speed > MAX_SPEED_PX: 
                scale = MAX_SPEED_PX / speed
                self.kf.statePost[2,0] = vx * scale
                self.kf.statePost[3,0] = vy * scale
            
            x = float(self.kf.statePost[0,0]) #float rather than int to maintain precision
            y = float(self.kf.statePost[1,0])
            self.centroids.append((x,y))
            self.missed = 0
            self.last_seen +=1 #update age of detection
        else: #no detection 
            #self.centroids.append(self.centroids[-1])
            px, py = self.kf.statePost[0,0], self.kf.statePost[1,0] 
            self.centroids.append((float(px), float(py))) # have velocity persist through brief occlusion if necessary 
            self.missed += 1
            self.last_seen +=1 #needs to also increment when missed

    def speed(self): 
        # utilizes kalman velocity
        vx = self.kf.statePost[2,0]
        vy = self.kf.statePost[3,0]
        return float(np.hypot(vx, vy))

    @property
    def last_position(self): 
        return self.centroids[-1] # returns last position stored in centroid



