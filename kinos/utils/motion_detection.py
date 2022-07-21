import cv2

class MotionDetection:
    def __init__(self, min_movement_area=40, weight=0.6):
        self.weighted_avg = None
        self.min_movement_area = min_movement_area
        self.weight = weight
        
        
    def is_current_frame_moving(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, tuple([21, 21]), 0)

        if self.weighted_avg is None:
            self.weighted_avg = frame_gray.copy().astype("float")

        frameDelta = cv2.absdiff(frame_gray, cv2.convertScaleAbs(self.weighted_avg))
        cv2.accumulateWeighted(frame_gray, self.weighted_avg,  self.weight)

        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) >= self.min_movement_area:
                return True, self.weighted_avg

        return False, self.weighted_avg
        
        
        
    