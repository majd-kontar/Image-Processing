import time

import cv2
import numpy as np


class SizeDetection:
    def __init__(self):
        self.kernel = np.ones((3, 3))
        self.wp = 630
        self.hp = 891
        self.RATIO = 3

    def detect(self, frame):
        edges = self.get_canny(frame)
        warp = self.detect_ref(edges)
        detected = self.get_object_sizes(warp)
        cv2.imshow('frame', frame)
        cv2.imshow('edges', edges)
        cv2.imshow('detected', detected)
        cv2.waitKey(0)

    def detect_ref(self, canny):

        dilation = cv2.dilate(canny, self.kernel, iterations=3)
        thresh = cv2.erode(dilation, self.kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        approx = 0
        for i in contours:
            if cv2.contourArea(i) > area:
                area = cv2.contourArea(i)
                approx = cv2.approxPolyDP(i, 0.009 * cv2.arcLength(i, True), True)

        pt_a = [approx[0][0][0], approx[0][0][1]]
        pt_b = [approx[1][0][0], approx[1][0][1]]
        pt_c = [approx[2][0][0], approx[2][0][1]]
        pt_d = [approx[3][0][0], approx[3][0][1]]
        # print(pt_a, pt_b, pt_c, pt_d)
        input_pts = np.float32([pt_a, pt_b, pt_c, pt_d])
        output_pts = np.float32([[0, 0],
                                 [0, self.hp],
                                 [self.wp, self.hp],
                                 [self.wp, 0]])
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        warp = cv2.warpPerspective(frame, M, (self.wp, self.hp), flags=cv2.INTER_LINEAR)
        return warp

    def get_object_sizes(self, warp):
        canny = self.get_canny(warp)
        dilation = cv2.dilate(canny, self.kernel, iterations=3)
        thresh = cv2.erode(dilation, self.kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for i in contours:
            pts = []
            area = cv2.contourArea(i)
            if area > 10000:
                approx = cv2.approxPolyDP(i, 0.009 * cv2.arcLength(i, True), True)
                for j in approx:
                    pts.append([j[0][0], j[0][1]])
                objects.append([pts])
                cv2.arrowedLine(warp, pts[0], pts[1],
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(warp, pts[0], pts[4],
                                (255, 0, 255), 3, 8, 0, 0.05)
                w = self.find_dist(pts[1], pts[0])
                h = self.find_dist(pts[4], pts[0])
                print('Size: (%f, %f)' % (w, h))
                cv2.putText(warp, '{:.2f}cm'.format(w), (pts[1][0] + 5, pts[1][1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 0, 255), 1)
                cv2.putText(warp, '{:.2f}cm'.format(h), (pts[4][0] + 5, pts[4][1] + 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 0, 255), 1)
        # print(objects)
        return warp

    def find_dist(self, pts1, pts2):
        return ((pts2[0] // (self.RATIO * 10) - pts1[0] // (self.RATIO * 10)) ** 2 + (
                pts2[1] // (self.RATIO * 10) - pts1[1] // (self.RATIO * 10)) ** 2) ** 0.5

    def get_canny(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        canny = cv2.absdiff(dilate, thresh)
        return canny


if __name__ == '__main__':
    size_detection = SizeDetection()
    frame = cv2.imread('../resources/size.jpg')
    size_detection.detect(frame)
