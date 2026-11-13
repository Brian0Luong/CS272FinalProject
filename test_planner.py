import numpy as np

MAINTAIN = 0
ACCELERATE = 1
BRAKE = 2
CHANGE_LANE = 3


def planner(obs, reply):
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
        if kind == "unavailable":
            return MAINTAIN

        target_y = target_lane * 4.0

        # close enough to target
        if abs(ego_y - target_y) < 0.5:
            if ego_speed < 20:
                return ACCELERATE
            if ego_speed > 30:
                return BRAKE
            return MAINTAIN

        if current_lane != target_lane:
            return CHANGE_LANE

        return MAINTAIN