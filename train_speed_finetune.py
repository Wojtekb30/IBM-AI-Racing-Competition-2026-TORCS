import os
import sys
import gym
import numpy as np
from gym import spaces
from gym_torcs import TorcsEnv
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit


BASE_MODEL_NAME = "td3_torcs_centerfinish_v2"
FINETUNE_MODEL_NAME = "td3_torcs_centerfinish_v4"
LEARN_CHUNK_TIMESTEPS = 15000


class TorcsSpeedFineTuneEnv(gym.Env):
    def __init__(self, vision=False, throttle=True, action_repeat=2):
        super(TorcsSpeedFineTuneEnv, self).__init__()
        self.env = TorcsEnv(vision=vision, throttle=throttle, gear_change=False)
        self.env.gear_change = True
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_repeat = action_repeat

        self.env.terminal_judge_start = 7000
        self.env.termination_limit_progress = 0.1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(76,),
            dtype=np.float32,
        )

        self.reset_count = 0
        self.stuck_steps = 0
        self.prev_damage = 0.0
        self.prev_steer = 0.0
        self.prev_steer_delta = 0.0
        self.prev_track_pos = 0.0
        self.prev_angle = 0.0
        self.prev_curve_phase = 0.0
        self.prev_dist_raced = 0.0
        self.prev_speed = 0.0

    def _auto_gear(self, speed):
        if speed < 38.0:
            return 1.0
        if speed < 70.0:
            return 2.0
        if speed < 102.0:
            return 3.0
        if speed < 135.0:
            return 4.0
        if speed < 170.0:
            return 5.0
        return 6.0

    def _as_vector(self, value, size):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= size:
            return arr[:size]
        out = np.zeros(size, dtype=np.float32)
        out[: arr.size] = arr
        return out

    def _build_observation(self):
        raw_obs = self.env.client.S.d
        track = self._as_vector(raw_obs.get("track", []), 19)
        focus = self._as_vector(raw_obs.get("focus", []), 5)
        wheel_spin = self._as_vector(raw_obs.get("wheelSpinVel", []), 4)
        opponents = self._as_vector(raw_obs.get("opponents", []), 36)

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

    def _safe_action(self, action):
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        steer = float(np.clip(act[0], -1.0, 1.0))
        throttle = float(np.clip((act[1] + 1.0) / 2.0, 0.0, 1.0))

        raw_obs = self.env.client.S.d
        track = self._as_vector(raw_obs.get("track", []), 19)
        front_dist = float(np.max(track[8:11]))
        abs_angle = abs(float(raw_obs.get("angle", 0.0)))
        track_pos_signed = float(raw_obs.get("trackPos", 0.0))
        track_pos = abs(track_pos_signed)
        speed = float(raw_obs.get("speedX", 0.0))
        left_open = float(np.mean(track[:5]))
        right_open = float(np.mean(track[14:]))
        far_left_open = float(np.mean(track[:3]))
        far_right_open = float(np.mean(track[16:]))

        curve_asym = abs(left_open - right_open) / 70.0
        curve_entry = max(0.0, (52.0 - front_dist) / 52.0)
        heading_risk = abs_angle / 0.45
        entry_bias = abs(far_left_open - far_right_open) / 75.0

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
        steer = float(
            np.clip(steer, self.prev_steer - max_steer_delta, self.prev_steer + max_steer_delta)
        )
        if straightness > 0.72 and track_pos < 0.58 and abs_angle < 0.27:
            steer = float(np.clip((0.90 * self.prev_steer) + (0.10 * steer), -0.28, 0.28))
        elif straightness > 0.55:
            steer = float(np.clip((0.84 * self.prev_steer) + (0.16 * steer), -0.36, 0.36))

        if straightness > 0.78 and abs(track_pos_signed) < 0.18 and abs_angle < 0.08:
            steer = 0.35 * steer
        elif straightness > 0.62 and abs(track_pos_signed) < 0.30 and abs_angle < 0.13:
            steer = 0.60 * steer

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

        gear = self._auto_gear(speed)

        return np.array([steer, throttle, gear], dtype=np.float32)

    def _shape_reward(self, env_done, action):
        raw_obs = self.env.client.S.d
        track = self._as_vector(raw_obs.get("track", []), 19)
        steer = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))

        speed = float(raw_obs.get("speedX", 0.0))
        angle = float(raw_obs.get("angle", 0.0))
        track_pos_signed = float(raw_obs.get("trackPos", 0.0))
        track_pos = abs(track_pos_signed)
        front_dist = float(np.max(track[8:11]))
        damage = float(raw_obs.get("damage", 0.0))
        rpm = float(raw_obs.get("rpm", 0.0))
        dist_raced = float(raw_obs.get("distRaced", 0.0))
        left_open = float(np.mean(track[:5]))
        right_open = float(np.mean(track[14:]))

        far_left_open = float(np.mean(track[:3]))
        far_right_open = float(np.mean(track[16:]))
        curve_asym = abs(left_open - right_open) / 70.0
        curve_entry = max(0.0, (52.0 - front_dist) / 52.0)
        heading_risk = abs(angle) / 0.45
        entry_bias = abs(far_left_open - far_right_open) / 75.0

        curve_phase = float(
            np.clip(0.85 * curve_asym + 1.40 * curve_entry + 0.75 * heading_risk + 0.50 * entry_bias, 0.0, 2.4)
        )
        straightness = float(
            np.clip(1.0 - 0.75 * curve_asym - 1.00 * curve_entry - 0.65 * heading_risk, 0.0, 1.0)
        )
        curve_relax = float(np.clip(1.0 - 0.9 * curve_phase, 0.0, 1.0))
        curve_phase_rise = max(0.0, curve_phase - self.prev_curve_phase)
        self.prev_curve_phase = curve_phase

        progress = speed * np.cos(angle)
        dist_delta = max(0.0, dist_raced - self.prev_dist_raced)
        self.prev_dist_raced = dist_raced
        speed_drop = max(0.0, self.prev_speed - speed)
        self.prev_speed = speed
        safe_speed = min(185.0, np.sqrt(max(front_dist, 1.0)) * 16.0)
        speed_ratio = speed / max(safe_speed, 1.0)
        straight_target_speed = 86.0 + (26.0 * straightness)
        under_speed_ratio = max(0.0, (straight_target_speed - speed) / max(straight_target_speed, 1.0))
        curve_target_speed = 92.0 - (38.0 * min(curve_phase, 1.0))
        curve_overspeed = max(0.0, speed - curve_target_speed)

        progress_term = float(np.clip(progress / 35.0, -1.8, 3.0))
        lap_progress_bonus = 3.6 * float(np.clip(dist_delta, 0.0, 4.0))
        survival_bonus = 0.30
        speed_bonus = float(np.clip((speed - 78.0) / 85.0, 0.0, 0.5)) * float(
            np.clip(front_dist / 48.0, 0.15, 1.0)
        )
        straight_speed_bonus = straightness * float(np.clip((speed - 100.0) / 55.0, 0.0, 0.25))
        speed_band_center = 105.0
        speed_band_reward = straightness * 0.12 * max(0.0, 1.0 - (abs(speed - speed_band_center) / 18.0))
        curve_control_bonus = (1.0 - straightness) * (1.0 - min(track_pos, 1.0)) * (
            1.0 - min(abs(angle) / 0.9, 1.0)
        )
        curve_brake_zone = float(np.clip((curve_phase - 0.12) / 1.00, 0.0, 1.0))
        curve_target_ratio = 0.48
        curve_speed_match_bonus = curve_brake_zone * max(
            0.0, 1.0 - (abs(speed_ratio - curve_target_ratio) / 0.22)
        )
        early_brake_bonus = curve_brake_zone * float(np.clip(speed_drop / 7.0, 0.0, 1.6)) * float(
            np.clip(0.55 + 2.2 * curve_phase_rise, 0.0, 1.8)
        )
        curve_late_brake_penalty = curve_brake_zone * max(0.0, speed_ratio - 0.58) * 4.0
        no_decel_penalty = curve_brake_zone * max(0.0, 0.55 - (speed_drop / 6.0)) * float(
            np.clip((speed - 65.0) / 30.0, 0.0, 1.0)
        )

        rpm_quality = 1.0 - min(abs(rpm - 7600.0) / 7600.0, 1.0)
        rpm_efficiency_bonus = straightness * float(np.clip(speed / 114.0, 0.0, 0.6)) * rpm_quality

        center_quality = max(0.0, 1.0 - track_pos)
        center_term = 0.8 * (center_quality ** 1.2)
        heading_term = 1.2 * (1.0 - min(abs(angle) / 0.75, 1.0))
        straight_hold_bonus = straightness * float(np.clip(speed / 125.0, 0.0, 1.2)) * max(0.0, 1.0 - (2.8 * abs(steer)))
        center_stability_bonus = 0.5 * float(np.clip(speed / 120.0, 0.0, 1.0)) * (center_quality ** 1.2)

        track_pos_delta = abs(track_pos - self.prev_track_pos)
        angle_delta = abs(angle - self.prev_angle)
        steer_delta = steer - self.prev_steer
        steer_jerk = steer_delta - self.prev_steer_delta
        steer_penalty = (0.9 + 1.6 * straightness) * abs(steer)
        steer_delta_penalty = (2.2 + 8.2 * straightness) * abs(steer_delta)
        steer_jerk_penalty = (1.5 + 6.0 * straightness) * abs(steer_jerk)
        oscillation_penalty = 0.0
        if steer_delta * self.prev_steer_delta < -0.0004 and abs(steer_delta) > 0.012:
            osc_gate = 1.0 if (straightness > 0.30 and track_pos < 0.75 and abs(angle) < 0.35) else 0.7
            oscillation_penalty = osc_gate * (3.2 + 5.4 * straightness) * min(1.0, abs(steer_delta) + abs(self.prev_steer_delta))
        flicker_penalty = (0.8 + 2.0 * straightness) * abs(steer_delta) * abs(self.prev_steer_delta)
        center_drift_penalty = (3.0 + 4.4 * straightness) * track_pos_delta
        heading_change_penalty = (1.6 + 3.0 * straightness) * angle_delta
        off_center_penalty = (2.4 + 3.4 * straightness) * max(0.0, track_pos - 0.55)
        severe_off_center_penalty = (10.0 + 6.0 * straightness) * max(0.0, track_pos - 0.78)
        border_penalty = 9.0 * max(0.0, track_pos - 0.65) ** 2

        under_speed_penalty = (straightness ** 2) * curve_relax * 0.25 * under_speed_ratio
        speed_low_penalty = straightness * 0.20 * max(0.0, (92.0 - speed) / 45.0)
        speed_high_penalty = straightness * 0.20 * max(0.0, (speed - 120.0) / 35.0)
        lugging_penalty = straightness * float(np.clip(speed / 145.0, 0.0, 1.0)) * max(0.0, (4300.0 - rpm) / 4300.0)
        redline_penalty = max(0.0, (rpm - 9600.0) / 2400.0)

        overspeed_penalty = (1.8 + 9.0 * curve_phase) * (max(0.0, speed_ratio - 1.0) ** 2)
        curve_overspeed_penalty = (2.2 + 4.2 * curve_brake_zone) * ((curve_overspeed / 8.0) ** 2)
        edge_penalty = (4.2 + 5.6 * straightness) * max(0.0, track_pos - 0.60)
        high_speed_edge_penalty = float(np.clip(speed / 130.0, 0.0, 1.4)) * 12.0 * max(0.0, track_pos - 0.54)
        wall_penalty = (6.0 + 6.0 * curve_phase) * max(0.0, (24.0 - front_dist) / 24.0)

        damage_delta = max(0.0, damage - self.prev_damage)
        self.prev_damage = damage
        damage_penalty = min(60.0, 0.18 * damage_delta)

        reward = (
            progress_term
            + survival_bonus
            + lap_progress_bonus
            + speed_bonus
            + straight_speed_bonus
            + speed_band_reward
            + curve_control_bonus
            + curve_speed_match_bonus
            + early_brake_bonus
            + rpm_efficiency_bonus
            + center_term
            + heading_term
            + straight_hold_bonus
            + center_stability_bonus
            - steer_penalty
            - steer_delta_penalty
            - steer_jerk_penalty
            - oscillation_penalty
            - flicker_penalty
            - center_drift_penalty
            - heading_change_penalty
            - off_center_penalty
            - severe_off_center_penalty
            - border_penalty
            - under_speed_penalty
            - curve_late_brake_penalty
            - no_decel_penalty
            - speed_low_penalty
            - speed_high_penalty
            - lugging_penalty
            - redline_penalty
            - overspeed_penalty
            - curve_overspeed_penalty
            - edge_penalty
            - high_speed_edge_penalty
            - wall_penalty
            - damage_penalty
        )

        self.prev_steer = steer
        self.prev_steer_delta = steer_delta
        self.prev_track_pos = track_pos
        self.prev_angle = angle

        if progress < 3.0:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        force_done = False
        if self.stuck_steps > 80:
            reward -= 12.0
        if self.stuck_steps > 160:
            reward -= 28.0
            force_done = True

        if damage_delta > 5.0:
            reward -= 32.0
            force_done = True

        if track_pos > 1.00:
            reward -= 50.0
            force_done = True

        if np.cos(angle) < -0.1:
            reward -= 10.0
            force_done = True

        if env_done:
            reward -= 30.0

        return float(np.clip(reward, -25.0, 25.0)), force_done

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        safe_action = self._safe_action(action)

        for _ in range(self.action_repeat):
            _, _, done, info = self.env.step(safe_action)
            shaped_reward, force_done = self._shape_reward(done, safe_action)
            total_reward += shaped_reward
            if force_done:
                done = True
            if done:
                break

        return self._build_observation(), total_reward, done, info

    def reset(self):
        self.reset_count += 1
        relaunch = self.reset_count % 5 == 0
        self.stuck_steps = 0
        self.prev_steer = 0.0
        self.prev_steer_delta = 0.0
        self.prev_track_pos = 0.0
        self.prev_angle = 0.0
        self.prev_curve_phase = 0.0
        self.prev_dist_raced = 0.0
        self.prev_speed = 0.0

        self.env.reset(relaunch=relaunch)
        self.prev_damage = float(self.env.client.S.d.get("damage", 0.0))
        self.prev_track_pos = abs(float(self.env.client.S.d.get("trackPos", 0.0)))
        self.prev_angle = float(self.env.client.S.d.get("angle", 0.0))
        self.prev_curve_phase = 0.0
        self.prev_dist_raced = float(self.env.client.S.d.get("distRaced", 0.0))
        self.prev_speed = float(self.env.client.S.d.get("speedX", 0.0))
        return self._build_observation()

    def close(self):
        self.env.end()


def load_finetune_model(env, action_noise):
    finetune_path = FINETUNE_MODEL_NAME + ".zip"
    finetune_replay = FINETUNE_MODEL_NAME + "_replay_buffer.pkl"
    base_path = BASE_MODEL_NAME + ".zip"
    base_replay = BASE_MODEL_NAME + "_replay_buffer.pkl"

    if os.path.exists(finetune_path):
        try:
            print("Resuming fine-tuned model:", FINETUNE_MODEL_NAME)
            model = TD3.load(finetune_path, env=None, action_noise=action_noise)
        except Exception as err:
            print("Could not load fine-tuned model:", err)
            sys.exit(1)
    else:
        if not os.path.exists(base_path):
            print("Base model not found:", base_path)
            print("Please place", BASE_MODEL_NAME, "before running this fine-tune.")
            sys.exit(1)
        try:
            print("Loading base model for fine-tuning:", BASE_MODEL_NAME)
            model = TD3.load(base_path, env=None, action_noise=action_noise)
        except Exception as err:
            print("Could not load base model:", err)
            sys.exit(1)

    model.set_env(env)
    model.action_noise = action_noise

    replay_path = finetune_replay if os.path.exists(finetune_replay) else base_replay
    if os.path.exists(replay_path):
        try:
            model.load_replay_buffer(replay_path)
            print("Loaded replay buffer:", replay_path)
        except Exception as err:
            print("Could not load replay buffer:", err)

    return model


if __name__ == "__main__":
    env = None
    action_noise = NormalActionNoise(
        mean=np.zeros(2),
        sigma=np.array([0.006, 0.010], dtype=np.float32),
    )

    print("Preparing speed fine-tuning...")
    env = TorcsSpeedFineTuneEnv(vision=False, throttle=True, action_repeat=2)
    model = load_finetune_model(env, action_noise)

    model.train_freq = TrainFreq(4, TrainFrequencyUnit.STEP)
    model.gradient_steps = 1

    print("Starting speed fine-tuning (Ctrl+C to stop and save)...")
    chunks_completed = 0
    try:
        while True:
            try:
                model.learn(
                    total_timesteps=LEARN_CHUNK_TIMESTEPS,
                    log_interval=10,
                    reset_num_timesteps=False,
                )
                chunks_completed += 1
                print("Completed fine-tune chunks:", chunks_completed)

                if chunks_completed % 2 == 0:
                    model.save(FINETUNE_MODEL_NAME)
                    model.save_replay_buffer(FINETUNE_MODEL_NAME + "_replay_buffer.pkl")
                    print("Periodic speed model save complete.")
            except Exception as err:
                print("Fine-tune chunk failed:", err)
                print("Rebuilding TORCS environment and continuing...")
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass

                env = TorcsSpeedFineTuneEnv(vision=False, throttle=True, action_repeat=2)
                model.set_env(env)
                model.action_noise = action_noise
                model.train_freq = TrainFreq(4, TrainFrequencyUnit.STEP)
                model.gradient_steps = 1
    except KeyboardInterrupt:
        print("Fine-tuning interrupted manually.")
    finally:
        if env is not None:
            env.close()
        model.save(FINETUNE_MODEL_NAME)
        model.save_replay_buffer(FINETUNE_MODEL_NAME + "_replay_buffer.pkl")
        print("Speed fine-tuned model and replay buffer saved.")
