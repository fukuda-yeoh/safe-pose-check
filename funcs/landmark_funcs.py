import cv2 as cv
import mediapipe as mp
import numpy as np

# MediaPipe DrawingStyleとDrawingSpecをインポート
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_landmarks_on_image(
    rgb_image, detection_result, color=(0, 255, 0)
):  # color引数を追加
    """
    検出されたポーズランドマークを画像に描画します。

    Args:
        rgb_image: ランドマークを描画するRGB画像。
        detection_result: MediaPipe PoseLandmarkerResultオブジェクト（pose_landmarksのリストを含む）。
        color: ランドマークを描画する色 (B, G, R) タプル。デフォルトは緑。
    Returns:
        ランドマークが描画された画像。
    """
    annotated_image = np.copy(rgb_image)

    # ランドマーク描画スタイルをカスタマイズするためのDrawingSpec
    # ここでcolor引数を使用
    landmark_drawing_spec = mp_drawing.DrawingSpec(
        color=color, thickness=2, circle_radius=2
    )
    connection_drawing_spec = mp_drawing.DrawingSpec(
        color=color, thickness=2, circle_radius=2
    )

    # detection_result が Mediapipe の PoseLandmarkerResult オブジェクトの場合
    if hasattr(detection_result, "pose_landmarks") and detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=mp_drawing_styles.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec,
            )
    # detection_result がランドマークリストの場合 (EventDetector内で単一人物として処理する場合など)
    elif isinstance(detection_result, list) and detection_result:
        for pose_landmarks in detection_result:  # リスト内の各人物（ここでは単一人物を想定）
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=mp_drawing_styles.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec,
            )
    return annotated_image


def get_landmark_labels(filepath):
    """
    指定されたファイルパスからランドマークのラベルを読み込み、辞書として返します。

    Args:
        filepath (Path): ランドマークラベルファイルのパス。

    Returns:
        dict: ランドマークのインデックスをキー、ラベル名を値とする辞書。
    """
    labels = {}
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            labels[line.strip()] = i
    return labels


def landmarks_to_array(pose_landmarks_list):
    """
    MediaPipeのLandmarkオブジェクトのリストからNumPy配列に変換します。
    単一の pose_landmarks オブジェクトを想定。

    Args:
        pose_landmarks_list (list): MediaPipeのNormalizedLandmarkオブジェクトのリストを含むリスト。
                                   （通常は [pose_landmarks_object] の形式）

    Returns:
        np.ndarray: 各ランドマークの [x, y, z, visibility, presence] を含むNumPy配列。
                    visibilityとpresenceがNoneの場合は0.0を埋めます。
    """
    if not pose_landmarks_list:
        return None

    # PoseLandmarkerResult の pose_landmarks はリストのリストになるため、最初の要素を取得
    # 例: result.pose_landmarks = [[lm1, lm2, ...]]
    # 複数人検出時は [[person1_lm1, ...], [person2_lm1, ...]]
    # 今回は num_poses=1 なので、pose_landmarks_list[0] が単一人物のランドマークリストとなる
    landmarks_data = []
    for landmark in pose_landmarks_list[0].landmark:
        landmarks_data.append(
            [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility if landmark.visibility is not None else 0.0,
                landmark.presence if landmark.presence is not None else 0.0,
            ]
        )
    return np.array(landmarks_data, dtype=np.float32)


def draw_arrow_on_image(image, start_point, end_point, color=(0, 0, 255), thickness=2):
    """
    画像上に矢印を描画します。

    Args:
        image (np.ndarray): 描画対象の画像。
        start_point (tuple): 矢印の開始座標 (x, y)。
        end_point (tuple): 矢印の終了座標 (x, y)。
        color (tuple): 矢印の色 (B, G, R)。
        thickness (int): 矢印の太さ。

    Returns:
        np.ndarray: 矢印が描画された画像。
    """
    # 座標を整数に変換
    # normalized_x * width, normalized_y * height
    start_point_px = (
        int(start_point[0] * image.shape[1]),
        int(start_point[1] * image.shape[0]),
    )
    end_point_px = (
        int(end_point[0] * image.shape[1]),
        int(end_point[1] * image.shape[0]),
    )

    return cv.arrowedLine(image, start_point_px, end_point_px, color, thickness)


def draw_roi_on_image(image, roi_points, color=(0, 255, 255), thickness=2):
    """
    画像上にROI（関心領域）を描画します。

    Args:
        image (np.ndarray): 描画対象の画像。
        roi_points (np.ndarray): ROIの頂点座標の配列 (正規化された座標)。
        color (tuple): ROIの線の色 (B, G, R)。
        thickness (int): ROIの線の太さ。

    Returns:
        np.ndarray: ROIが描画された画像。
    """
    h, w, _ = image.shape
    # 正規化された座標をピクセル座標に変換
    pts = (roi_points * np.array([w, h])).astype(np.int32)
    # 多角形を描画 (閉じる)
    return cv.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


def draw_clear_roi_on_image(image, clear_roi_points, color=(255, 0, 255), thickness=2):
    """
    画像上にクリアROI（退去領域）を描画します。

    Args:
        image (np.ndarray): 描画対象の画像。
        clear_roi_points (np.ndarray): クリアROIの頂点座標の配列 (正規化された座標)。
        color (tuple): クリアROIの線の色 (B, G, R)。
        thickness (int): クリアROIの線の太さ。

    Returns:
        np.ndarray: クリアROIが描画された画像。
    """
    h, w, _ = image.shape
    # 正規化された座標をピクセル座標に変換
    pts = (clear_roi_points * np.array([w, h])).astype(np.int32)
    # 多角形を描画 (閉じる)
    return cv.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
