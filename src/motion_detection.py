import cv2
import numpy as np


class MotionDetector:
    def __init__(self):
        self.prev = None
        self.count = 5

    def save_ref_frame(self, ref, name):
        cv2.imwrite('../resources/' + name + '.jpeg', ref)

    def detect_ref_frame(self, ref, frame):
        difference = cv2.absdiff(ref, frame)
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        threshold = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)[1]
        dilate = cv2.dilate(threshold, None, iterations=15)
        summation = np.sum(dilate == 255)
        contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dilate = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
        if summation > 10000:
            dilate = cv2.putText(dilate, 'Motion Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 10000:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.putText(frame, 'Motion Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        cv2.imshow('Ref', dilate)
        cv2.waitKey(1)
        return dilate

    def detect_prev_image(self, frame, count=5):
        if self.count == count:
            self.prev = frame.copy()
            self.count = 0
        difference = cv2.absdiff(self.prev, frame)
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        threshold = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)[1]
        dilate = cv2.dilate(threshold, None, iterations=2)
        sum = np.sum(dilate == 255)
        dilate = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
        if sum > 10000:
            dilate = cv2.putText(dilate, 'Motion Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Prev', dilate)
        cv2.waitKey(1)
        self.count += 1
        return dilate


if __name__ == '__main__':
    motion_detector = MotionDetector()
    cam = cv2.VideoCapture('../resources/motion.mp4')
    # input('Press to capture background')
    # _, ref = cam.read()
    # cv2.imshow('Background', ref)
    # cv2.waitKey(0)
    # motion_detector.save_ref_frame(frame, 'background')
    ref = cv2.imread('../resources/background.jpeg')
    while True:
        grabbed, frame = cam.read()
        if grabbed:
            motion_detector.detect_ref_frame(ref, frame)
            motion_detector.detect_prev_image(frame, 5)
        else:
            break
