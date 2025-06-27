import cv2
import numpy as np
from funcs import rotate_marker_corners

# Arucoの辞書を指定
dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

cameraMatrix = np.array(
    [
        [635.8283543606941, 0.0, 253.67538054709416],
        [0.0, 636.5077974749544, 320.7767679600789],
        [0.0, 0.0, 1.0],
    ]
)
distCoeffs = np.array(
    [
        -0.013855030664346785,
        0.036480883659782824,
        -0.004879021042728567,
        0.0019079058483680378,
        -0.21054703261551563,
    ]
)
# Webカメラを起動
cap = cv2.VideoCapture(1)
# 入力動画ファイルのパス
# video_path = r"C:\Users\ryono\SafePoseCheck\sample_data\Aruco.mp4"
# cap = cv2.VideoCapture(video_path)

# 表示サイズを設定
display_width = 2160
display_height = 3840

# ここにフルスクリーン表示のための行を追加
cv2.namedWindow("Aruco Marker Horizontal Vector", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Aruco Marker Horizontal Vector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    if not ret:
        break

    # Arucoマーカーの検出
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, dict_aruco)

    # 検出されたマーカーがあれば処理
    if ids is not None:
        # マーカーの枠を表示
        cv2.aruco.drawDetectedMarkers(frame, corners)
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.58, cameraMatrix, distCoeffs
        )

        markercorners, mrv = rotate_marker_corners(rvecs[0], 0.4, tvecs[0])
        # print(markercorners[0])
        # print(markercorners[1])
        print(markercorners[1] - markercorners[0])

        for i in range(ids.size):
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1)

    # フレームをリサイズ
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # リサイズしたフレームを表示
    cv2.imshow("Aruco Marker Horizontal Vector", resized_frame)

    # 'q' キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 終了処理
cap.release()
