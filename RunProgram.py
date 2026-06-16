import socket
import time
import numpy as np
from stable_baselines3 import TD3
print("Autorzy | Authors: Wojciech B, Patryk N")
print("Druzyna | Team: WIPy_z_Polsl")

HOST = "localhost"
PORT = 3001
CLIENT_ID = "SCR"
MODEL_PATH = "td3_torcs_centerfinish_v2"
SOCKET_TIMEOUT_SEC = 1.0

TRACK_ANGLES = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
FOCUS_VALUES = "-90 -45 0 45 90"
DATA_SIZE = 2 ** 17
PREDICT_TIMEOUT_SEC = 0.12
MAX_PREDICT_TIMEOUT_STEPS = 4
AI_RECOVERY_STEPS = 18


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


def clip(value, low, high):
    return max(low, min(high, value))


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


class DriveGuard:
    def __init__(self):
        self.mode = "AI"
        self.predict_timeout_steps = 0
        self.ai_recovery_steps = 0
        self.prev_steer = 0.0

    def _set_mode(self, new_mode, reason):
        if self.mode != new_mode:
            print("Mode switch:", self.mode, "->", new_mode, "|", reason)
            self.mode = new_mode

    def _auto_gear(self, speed):
        if speed < 38.0:
            return 1
        if speed < 70.0:
            return 2
        if speed < 102.0:
            return 3
        if speed < 135.0:
            return 4
        if speed < 170.0:
            return 5
        return 6

    def _deterministic_action(self, raw_obs):
        track = as_vector(raw_obs.get("track", []), 19)
        speed = float(raw_obs.get("speedX", 0.0))
        angle = float(raw_obs.get("angle", 0.0))
        track_pos = float(raw_obs.get("trackPos", 0.0))
        front_dist = float(np.max(track[8:11]))

        steer = clip((angle * 0.95) - (track_pos * 0.85), -1.0, 1.0)
        steer = clip((0.78 * self.prev_steer) + (0.22 * steer), -1.0, 1.0)

        safe_speed = min(165.0, np.sqrt(max(front_dist, 1.0)) * 15.5)
        throttle = 0.34
        brake = 0.0

        if speed > safe_speed:
            throttle = 0.0
            brake = clip((speed - safe_speed) / 28.0, 0.0, 1.0)
        elif speed < safe_speed * 0.9:
            throttle = 0.46

        if abs(track_pos) > 0.90:
            throttle = min(throttle, 0.24)

        self.prev_steer = steer
        return steer, throttle, brake, self._auto_gear(speed)

    def _safe_ai_action(self, action, raw_obs):
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        steer = float(np.clip(act[0], -1.0, 1.0))
        throttle = float(np.clip((act[1] + 1.0) / 2.0, 0.0, 1.0))

        track = as_vector(raw_obs.get("track", []), 19)
        front_dist = float(np.max(track[8:11]))
        abs_angle = abs(float(raw_obs.get("angle", 0.0)))
        track_pos_signed = float(raw_obs.get("trackPos", 0.0))
        track_pos = abs(track_pos_signed)
        speed = float(raw_obs.get("speedX", 0.0))
        left_open = float(np.mean(track[:5]))
        right_open = float(np.mean(track[14:]))
        curve_asym = abs(left_open - right_open) / 70.0
        curve_entry = max(0.0, (52.0 - front_dist) / 52.0)
        heading_risk = abs_angle / 0.45
        entry_bias = abs(float(np.mean(track[:3])) - float(np.mean(track[16:]))) / 75.0

        curve_risk = float(
            np.clip(0.85 * curve_asym + 1.40 * curve_entry + 0.75 * heading_risk + 0.50 * entry_bias, 0.0, 2.4)
        )
        straightness = float(
            np.clip(1.0 - 0.75 * curve_asym - 1.00 * curve_entry - 0.65 * heading_risk, 0.0, 1.0)
        )

        throttle_cap = float(np.clip(1.0 - (0.90 * curve_risk), 0.02, 0.92))
        throttle = min(throttle, throttle_cap)

        safe_speed_local = min(175.0, np.sqrt(max(front_dist, 1.0)) * 15.5)
        if curve_risk > 0.25 and speed > safe_speed_local:
            throttle = min(throttle, max(0.0, 0.18 - 0.009 * (speed - safe_speed_local)))

        max_steer_delta = 0.32 - (0.22 * straightness)
        steer = clip(steer, self.prev_steer - max_steer_delta, self.prev_steer + max_steer_delta)
        if straightness > 0.72 and track_pos < 0.58 and abs_angle < 0.27:
            steer = clip((0.90 * self.prev_steer) + (0.10 * steer), -0.28, 0.28)
        elif straightness > 0.55:
            steer = clip((0.84 * self.prev_steer) + (0.16 * steer), -0.36, 0.36)

        if track_pos > 0.78 and abs_angle < 0.20:
            throttle = min(throttle, 0.32)
        if track_pos_signed > 0.78 and steer > 0.0:
            steer *= 0.55
        elif track_pos_signed < -0.78 and steer < 0.0:
            steer *= 0.55
        if speed > 85.0 and straightness > 0.55:
            steer *= 0.75

        if straightness > 0.88 and track_pos < 0.32 and abs_angle < 0.10 and speed < 110.0:
            throttle = max(throttle, 0.52)
        elif straightness > 0.72 and track_pos < 0.50 and abs_angle < 0.18 and speed < 100.0:
            throttle = max(throttle, 0.40)

        if curve_risk > 0.70:
            throttle = min(throttle, 0.18)
        if curve_risk > 0.95:
            throttle = min(throttle, 0.08)

        if speed < 10.0 and curve_risk < 0.40 and track_pos < 0.50:
            throttle = max(throttle, 0.32)

        brake = 0.0
        safe_speed = min(185.0, np.sqrt(max(front_dist, 1.0)) * 16.0)
        if speed > safe_speed:
            brake = min(1.0, (speed - safe_speed) / 35.0)
            throttle = min(throttle, 0.2)

        self.prev_steer = steer
        return steer, throttle, brake, self._auto_gear(speed)

    def choose_action(self, model, obs, raw_obs):
        track_pos = abs(float(raw_obs.get("trackPos", 0.0)))
        off_track = track_pos > 1.0

        action = None
        predict_ok = False
        t0 = time.time()
        try:
            action, _ = model.predict(obs, deterministic=True)
            predict_ok = True
        except Exception:
            predict_ok = False
        elapsed = time.time() - t0

        if (not predict_ok) or (elapsed > PREDICT_TIMEOUT_SEC):
            self.predict_timeout_steps += 1
        else:
            self.predict_timeout_steps = 0

        if off_track:
            self._set_mode("DET", "off-track recovery")
            self.ai_recovery_steps = 0
        elif self.predict_timeout_steps >= MAX_PREDICT_TIMEOUT_STEPS:
            self._set_mode("DET", "AI output timeout")
            self.ai_recovery_steps = 0

        if self.mode == "AI" and predict_ok:
            return self._safe_ai_action(action, raw_obs)

        det_action = self._deterministic_action(raw_obs)

        if predict_ok and self.predict_timeout_steps == 0 and track_pos < 0.75 and abs(float(raw_obs.get("angle", 0.0))) < 0.35:
            self.ai_recovery_steps += 1
            if self.ai_recovery_steps >= AI_RECOVERY_STEPS:
                self._set_mode("AI", "recovered")
                self.ai_recovery_steps = 0
        else:
            self.ai_recovery_steps = 0

        return det_action


def action_message(steer, accel, brake, gear):
    return (
        "(accel %.3f)(brake %.3f)(gear %d)(steer %.3f)(clutch 0)(focus %s)(meta 0)"
        % (accel, brake, gear, steer, FOCUS_VALUES)
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
    drive_guard = DriveGuard()

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
            steer, accel, brake, gear = drive_guard.choose_action(model, obs, state)

            msg = action_message(steer, accel, brake, gear)
            sock.sendto(msg.encode("utf-8"), (HOST, PORT))
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        sock.close()
