import time
from queue import Queue
from threading import Thread

import cv2 as cv
import numpy as np  # numpyをインポート


class CameraThread(Thread):
    # cam_matrix と dist_coeffs をNoneをデフォルトとするオプション引数に変更
    def __init__(self, camera_no, cam_matrix=None, dist_coeffs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_no = camera_no
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.queue = Queue(maxsize=30)

        self.stop_flag = False
        self.cap = None
        self.start_time = time.time()

    def run(self):
        self.cap = cv.VideoCapture(self.camera_no)
        # カメラが開けない場合の確認を追加
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_no}.")
            self.stop_flag = True  # 開けなければスレッドを停止
            return

        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:  # フレームが読み込めなかった場合
                print(
                    f"Warning: Could not read frame from camera {self.camera_no}. Trying again..."
                )
                time.sleep(0.1)  # 少し待ってから再試行
                continue

            timestamp = int((time.time() - self.start_time) * 1000)

            # cam_matrix と dist_coeffs がNoneでない場合にのみ歪み補正を実行
            if self.cam_matrix is not None and self.dist_coeffs is not None:
                try:
                    undistorted_frame = cv.undistort(
                        frame, self.cam_matrix, self.dist_coeffs
                    )
                except cv.error as e:
                    print(f"Error during undistortion: {e}. Using original frame.")
                    undistorted_frame = frame  # エラー時は元のフレームを使用
            else:
                undistorted_frame = frame  # 補正パラメータがない場合は元のフレームを使用

            self.queue.put((timestamp, undistorted_frame))
        self.cap.release()

    def stop(self):
        self.stop_flag = True


if __name__ == "__main__":
    cam_no = 0
    # 歪み補正を無効にするため、cam_matrix と dist_coeffs を渡さない
    camera_thread = CameraThread(cam_no)
    camera_thread.start()

    # スレッドが開始するまで少し待つ
    time.sleep(1)

    while True:
        # queueが空でないことを確認してからgetを呼び出す
        if not camera_thread.queue.empty():
            t, raw_frame = camera_thread.queue.get()

            cv.imshow(f"Camera {cam_no}", raw_frame)
        else:
            # キューが空の場合は少し待つ
            time.sleep(0.01)

        key = cv.waitKey(1)
        if key == ord("q"):
            camera_thread.stop()
            break
        elif key == -1 and not camera_thread.is_alive():  # スレッドが停止していてキー入力がない場合
            print("Camera thread stopped unexpectedly.")
            break

    # メインループ終了後にスレッドが確実に停止しているか確認
    if camera_thread.is_alive():
        camera_thread.stop()
    camera_thread.join()  # スレッドの終了を待つ
    cv.destroyAllWindows()
