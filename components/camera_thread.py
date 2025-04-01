import time
from queue import Queue
from threading import Thread

import cv2 as cv


class CameraThread(Thread):
    def __init__(self, camera_no, cam_matrix, dist_coeffs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_no = camera_no
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.queue = Queue(maxsize=30)

        self.stop_flag = False
        self.cap = None
        self.frame = None
        self.start_time = time.time()

    def run(self):
        self.cap = cv.VideoCapture(self.camera_no)
        while not self.stop_flag:
            ret, frame = self.cap.read()
            timestamp = int((time.time() - self.start_time) * 1000)
            undistorted_frame = cv.undistort(frame, self.cam_matrix, self.dist_coeffs)  # 歪み補正
            if ret:
                self.queue.put((timestamp, undistorted_frame))
        self.cap.release()

    def stop(self):
        self.stop_flag = True


if __name__ == "__main__":
    cam_no = 0
    camera_thread = CameraThread(cam_no)
    camera_thread.start()

    while True:
        t, raw_frame = camera_thread.queue.get()

        cv.imshow(f"Camera {cam_no}", raw_frame)
        if cv.waitKey(1) == ord("q"):
            camera_thread.stop()
            break

    camera_thread.stop()
