import cv2


class EdgeDetector:
    def get_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blur, 40, 120)
        cv2.imshow('Edges', canny)
        cv2.waitKey(0)


if __name__ == '__main__':
    edge_detector = EdgeDetector()
    frame = cv2.imread('../resources/edge.jpg')
    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
    edge_detector.get_edges(frame)
