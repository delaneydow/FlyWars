from collections import dequeue


# === TRACKING CLASS ===
# used in tracking.py
class Track: 
    def __init__(self, track_id, centroid): 
        self.id = track_id
        self.centroids = deque(maxlen=HISTORY)
        self.centroids.append(centroid)
        self.missed = 0 # num of consecutive frames w/o detection
        self.last_seen = 1 # num. of frames since last scene aka how old 

        # ADD KALMAN STATE
        # using initial vector [x, y, vx, vy]T
        self.kf = cv2.KalmanFilter(4,2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, DT, 0],
            [0, 1, 0, DT], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]], dtype=np.float32)

        self.kf.measurementMatrix=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) *1e-1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-0

        self.kf.statePre = np.array([[centroid[0]],
                                     [centroid[1]],
                                     [0],
                                     [0]], dtype=np.float32)

    def predict(self): 
        pred = self.kf.predict()
        return pred[0,0], pred[1,0]

    def update(self, detection=None):
        if detection is not None: 
            measured = np.array([[np.float32(detection[0])],
                                 [np.float32(detection[1])]])
            self.kf.correct(measured)
            
            x = int(self.kf.statePost[0,0])
            y = int(self.kf.statePost[1,0])
            self.centroids.append((x,y))
            self.missed = 0
            self.last_seen +=1 #update age of detection
        else: #no detection 
            self.centroids.append(self.centroids[-1])
            self.missed += 1
        # initialize velocity using first two detections
        if len(self.centroids) >= 2:
            vx = self.centroids[-1][0] - self.centroids[-2][0]
            vy = self.centroids[-1][1] - self.centroids[-2][1]
            self.kf.statePre[2,0] = vx
            self.kf.statePre[3,0] = vy

    def speed(self): 
        if len(self.centroids) < 2:
            return 0.0
        dx = self.centroids[-1][0] - self.centroids[-2][0]
        dy = self.centroids[-1][1] - self.centroids[-2][1]
        return np.hypot(dx, dy)

    @property
    def last_position(self): 
        return self.centroids[-1] # returns last position stored in centroid



