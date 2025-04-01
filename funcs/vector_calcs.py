import cv2
import numpy as np


def calculate_angle(p1, p2, p3):
    """角度を計算する関数"""

    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])  # ベクトル1
    v2 = np.array([p3[0] - p1[0], p3[1] - p1[1]])  # ベクトル2

    # 角度の大きさを内積を使って計算
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
    # 2次元での外積（z軸方向の成分）を使って回転方向を判定
    cross_product = (v1[0] * v2[1]) - (v1[1] * v2[0])
    # 外積の符号で回転方向を判定して角度の符号を決定
    if cross_product < 0:
        angle = -angle  # 時計回りなら負の角
    # 結果を度数法に変換して返す
    return np.degrees(angle)


def angle_between(vec1: np.array, vec2: np.array) -> np.array:
    """
    Calculate anticlockwise angle between two vectors

    :param vec1: vectors to calculate angle from
    :type vec1: 2d numpy array of vectors
    :param vec2: vectors to calculate angle from
    :type vec2: 2d numpy array of vectors
    :return: angles
    :rtype: 2d numpy array of angles
    """
    mag1 = np.linalg.norm(vec1, axis=1)
    mag2 = np.linalg.norm(vec2, axis=1)

    vec1_hat = vec1 / mag1.reshape(-1, 1)
    vec2_hat = vec2 / mag2.reshape(-1, 1)

    cross_product = np.cross(vec1_hat, vec2_hat)
    # dot_product = np.dot(vec1_hat, vec2_hat.T)
    dot_product = np.einsum("ij, ij->i", vec1_hat, vec2_hat)
    angle = np.degrees(np.arccos(dot_product))
    angle[cross_product < 0] = -angle[cross_product < 0]
    return angle


def paste(dst, src, x, y, a, b):
    """画像の上に画像を貼り付ける関数"""

    resized_img = cv2.resize(src, (a, b))

    if x >= dst.shape[1] or y >= dst.shape[0]:
        return
    w = (
        min(dst.shape[1] - x, resized_img.shape[1])
        if x >= 0
        else min(max(resized_img.shape[1] + x, 0), dst.shape[1])
    )
    h = (
        min(dst.shape[0] - y, resized_img.shape[0])
        if y >= 0
        else min(max(resized_img.shape[0] + y, 0), dst.shape[0])
    )
    u = 0 if x >= 0 else min(-x, resized_img.shape[1] - 1)
    v = 0 if y >= 0 else min(-y, resized_img.shape[0] - 1)
    px = max(x, 0)
    py = max(y, 0)
    roi_dst = dst[py : py + h, px : px + w]
    roi_resized = resized_img[v : v + h, u : u + w]
    roi_dst[:] = roi_resized
