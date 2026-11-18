import numpy as np

LANE_LEFT = 0
MAINTAIN = 1
LANE_RIGHT = 2
ACCELERATE = 3
BRAKE = 4

LIDAR_RANGE = 60.0

def planner(obs, reply, request_action=0):
    # decode observation
    ego_speed = obs[32] * 40
    current_lane = int(obs[33] * 2)
    target_lane = int(obs[34] * 2)
    ego_y = obs[35] * 8.0

    # expand reply
    kind = reply["kind"]
    neighbor = reply["neighbor"]
    data = reply["data"]
    bits = reply["bits"]

    # interpret intents
    if kind == "intent":
        if neighbor == "front" and data == "brake":
            return BRAKE
        if neighbor == "rear" and data == "accelerate":
            return BRAKE
    elif kind == "lidar":
        lidar = np.asarray(data, dtype=np.float32).reshape(16, 2) / 32767.0
        distances = lidar[:, 0]
        relative_speeds = lidar[:, 1]

        lidar_distance = distances[0] * LIDAR_RANGE
        lidar_speed = relative_speeds[0] * LIDAR_RANGE

        if neighbor == "front":
            if lidar_speed != 0:
                # low value means about to collide
                dist_speed_ratio = lidar_distance / abs(lidar_speed)

                if lidar_distance < 40.0 and dist_speed_ratio < 4:
                    return BRAKE

        elif neighbor == "rear":
            if lidar_speed != 0:
                # low value means about to collide
                dist_speed_ratio = lidar_distance / abs(lidar_speed)

                if lidar_distance < 40.0 and dist_speed_ratio < 4:
                    return MAINTAIN
                return ACCELERATE
    elif kind == "unavailable":
        return MAINTAIN

    target_y = target_lane * 4.0

    # close enough to target
    if abs(ego_y - target_y) < 0.5:
        if ego_speed < 20:
            return ACCELERATE
        if ego_speed > 30:
            return BRAKE
        return MAINTAIN

    if current_lane < target_lane:
        return LANE_RIGHT
    elif current_lane > target_lane:
        return LANE_LEFT

    return MAINTAIN