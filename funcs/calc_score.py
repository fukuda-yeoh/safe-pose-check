def calculate_score_l(angle, min_angle, max_angle):
    if angle <= min_angle:
        return 100
    elif min_angle < angle <= max_angle:
        return int((100 * (max_angle - angle)) / (max_angle - min_angle))
    elif angle > max_angle:
        return 0
    return 0


def calculate_score_r(angle, min_angle, max_angle):
    if angle <= min_angle:
        return 0
    elif min_angle < angle <= max_angle:
        return int((100 * (angle - min_angle)) / (max_angle - min_angle))
    elif angle > max_angle:
        return 100
    return 0
