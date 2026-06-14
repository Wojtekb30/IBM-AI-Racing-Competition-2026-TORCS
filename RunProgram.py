import socket
import time
import numpy as np
from stable_baselines3 import TD3


HOST = "localhost"
PORT = 3001
CLIENT_ID = "SCR"
MODEL_PATH = "td3_torcs_trackaware_v1"
SOCKET_TIMEOUT_SEC = 1.0

TRACK_ANGLES = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
FOCUS_VALUES = "-90 -45 0 45 90"
DATA_SIZE = 2 ** 17


def _to_float_list(values):
    out = []
    for value in values:
        try:
            out.append(float(value))
        except ValueError:
            out.append(0.0)
    return out


def parse_server_state(server_string):
    server_string = server_string.strip()
    if not server_string:
        return {}

    if server_string.startswith("(") and server_string.endswith(")"):
        server_string = server_string[1:-1]

    parsed = {}
    for chunk in server_string.split(")("):
        parts = chunk.split(" ")
        key = parts[0]
        vals = parts[1:]
        if not vals:
            parsed[key] = 0.0
            continue
        floats = _to_float_list(vals)
        parsed[key] = floats[0] if len(floats) == 1 else floats
    return parsed


def as_vector(value, size):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size >= size:
        return arr[:size]
    out = np.zeros(size, dtype=np.float32)
    out[:arr.size] = arr
    return out


def build_observation(raw_obs):
    track = as_vector(raw_obs.get("track", []), 19)
    focus = as_vector(raw_obs.get("focus", []), 5)
    wheel_spin = as_vector(raw_obs.get("wheelSpinVel", []), 4)
    opponents = as_vector(raw_obs.get("opponents", []), 36)

    speed_x = float(raw_obs.get("speedX", 0.0))
    speed_y = float(raw_obs.get("speedY", 0.0))
    speed_z = float(raw_obs.get("speedZ", 0.0))
    rpm = float(raw_obs.get("rpm", 0.0))
    track_pos = float(raw_obs.get("trackPos", 0.0))
    angle = float(raw_obs.get("angle", 0.0))

    ahead_max = float(np.max(track[8:11]))
    ahead_mean = float(np.mean(track[7:12]))
    curve_hint = float((np.mean(track[:5]) - np.mean(track[14:])) / 200.0)
    corridor_width = float((track[0] + track[-1]) / 200.0)

    obs = np.concatenate(
        [
            np.array(
                [
                    speed_x / 300.0,
                    speed_y / 100.0,
                    speed_z / 100.0,
                    rpm / 10000.0,
                    np.clip(track_pos, -2.0, 2.0) / 2.0,
                    np.clip(angle / np.pi, -1.0, 1.0),
                    np.sin(angle),
                    np.cos(angle),
                ],
                dtype=np.float32,
            ),
            np.clip(track / 200.0, -1.0, 1.5),
            np.clip(focus / 200.0, -1.0, 1.5),
            np.clip(wheel_spin / 100.0, -2.0, 4.0),
            np.clip(opponents / 200.0, 0.0, 1.5),
            np.array(
                [
                    ahead_max / 200.0,
                    ahead_mean / 200.0,
                    curve_hint,
                    corridor_width,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return obs.astype(np.float32)


def safe_action(action, raw_obs):
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    steer = float(np.clip(act[0], -1.0, 1.0))
    throttle = float(np.clip((act[1] + 1.0) / 2.0, 0.0, 1.0))

    track = as_vector(raw_obs.get("track", []), 19)
    front_dist = float(np.max(track[8:11]))
    abs_angle = abs(float(raw_obs.get("angle", 0.0)))
    speed = float(raw_obs.get("speedX", 0.0))

    danger = min(1.0, (abs_angle / 0.70) + max(0.0, (30.0 - front_dist) / 30.0))
    throttle_cap = max(0.15, 1.0 - (0.75 * danger))
    throttle = min(throttle, throttle_cap)

    if speed < 5.0 and front_dist > 20.0:
        throttle = max(throttle, 0.35)

    brake = 0.0
    safe_speed = min(220.0, np.sqrt(max(front_dist, 1.0)) * 22.0)
    if speed > safe_speed:
        brake = min(1.0, (speed - safe_speed) / 35.0)
        throttle = min(throttle, 0.2)

    return steer, throttle, brake


def action_message(steer, accel, brake):
    return (
        "(accel %.3f)(brake %.3f)(gear 1)(steer %.3f)(clutch 0)(focus %s)(meta 0)"
        % (accel, brake, steer, FOCUS_VALUES)
    )


def connect_client(sock):
    init_msg = "%s(init %s)" % (CLIENT_ID, TRACK_ANGLES)
    print("Connecting to TORCS at %s:%d..." % (HOST, PORT))
    while True:
        sock.sendto(init_msg.encode("utf-8"), (HOST, PORT))
        try:
            data, _ = sock.recvfrom(DATA_SIZE)
            text = data.decode("utf-8", errors="ignore")
            if "***identified***" in text:
                print("Connected.")
                return
        except socket.timeout:
            print("Waiting for TORCS server...")
            time.sleep(0.3)


if __name__ == "__main__":
    print("Loading model:", MODEL_PATH)
    model = TD3.load(MODEL_PATH)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(SOCKET_TIMEOUT_SEC)

    try:
        connect_client(sock)

        while True:
            try:
                data, _ = sock.recvfrom(DATA_SIZE)
            except socket.timeout:
                continue

            text = data.decode("utf-8", errors="ignore")

            if "***shutdown***" in text:
                print("TORCS requested shutdown.")
                break
            if "***restart***" in text:
                print("TORCS restarted race.")
                break
            if "***identified***" in text:
                continue
            if not text.startswith("("):
                continue

            state = parse_server_state(text)
            obs = build_observation(state)
            action, _ = model.predict(obs, deterministic=True)
            steer, accel, brake = safe_action(action, state)

            msg = action_message(steer, accel, brake)
            sock.sendto(msg.encode("utf-8"), (HOST, PORT))
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        sock.close()
