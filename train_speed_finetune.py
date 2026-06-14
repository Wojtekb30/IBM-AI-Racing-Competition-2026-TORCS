import os
import sys
import gym
import numpy as np
from gym import spaces
from gym_torcs import TorcsEnv
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit


BASE_MODEL_CANDIDATES = ["td3_torcs_speed_v1", "td3_torcs_trackaware_v1"]
FINETUNE_MODEL_NAME = "td3_torcs_speed_v2"
LEARN_CHUNK_TIMESTEPS = 12000


class TorcsSpeedFineTuneEnv(gym.Env):
    def __init__(self, vision=False, throttle=True, action_repeat=2):
        super(TorcsSpeedFineTuneEnv, self).__init__()
        self.env = TorcsEnv(vision=vision, throttle=throttle, gear_change=False)
        self.action_space = self.env.action_space
        self.action_repeat = action_repeat

        self.env.terminal_judge_start = 5000
        self.env.termination_limit_progress = 0.2

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
        track_pos = abs(float(raw_obs.get("trackPos", 0.0)))
        speed = float(raw_obs.get("speedX", 0.0))

        danger = (
            (abs_angle / 0.55)
            + max(0.0, (track_pos - 0.65) / 0.35)
            + max(0.0, (28.0 - front_dist) / 28.0)
        )
        danger = min(1.5, danger)
        throttle_cap = float(np.clip(1.0 - (0.60 * danger), 0.08, 1.0))
        throttle = min(throttle, throttle_cap)

        left_open = float(np.mean(track[:5]))
        right_open = float(np.mean(track[14:]))
        straightness = float(
            np.clip(front_dist / 85.0, 0.0, 1.0)
            * np.clip(1.0 - abs(left_open - right_open) / 65.0, 0.0, 1.0)
            * np.clip(1.0 - abs_angle / 0.35, 0.0, 1.0)
        )

        max_steer_delta = 0.40 - (0.22 * straightness)
        steer = float(
            np.clip(steer, self.prev_steer - max_steer_delta, self.prev_steer + max_steer_delta)
        )
        if straightness > 0.70 and track_pos < 0.55 and abs_angle < 0.25:
            steer = float(np.clip((0.75 * self.prev_steer) + (0.25 * steer), -0.40, 0.40))

        if speed < 8.0 and front_dist > 35.0 and track_pos < 0.55 and abs_angle < 0.25:
            throttle = max(throttle, 0.45)

        return np.array([steer, throttle], dtype=np.float32)

    def _shape_reward(self, env_done, action):
        raw_obs = self.env.client.S.d
        track = self._as_vector(raw_obs.get("track", []), 19)
        steer = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))

        speed = float(raw_obs.get("speedX", 0.0))
        angle = float(raw_obs.get("angle", 0.0))
        track_pos = abs(float(raw_obs.get("trackPos", 0.0)))
        front_dist = float(np.max(track[8:11]))
        damage = float(raw_obs.get("damage", 0.0))
        left_open = float(np.mean(track[:5]))
        right_open = float(np.mean(track[14:]))

        straightness = float(
            np.clip(front_dist / 90.0, 0.0, 1.0)
            * np.clip(1.0 - abs(left_open - right_open) / 70.0, 0.0, 1.0)
            * np.clip(1.0 - abs(angle) / 0.35, 0.0, 1.0)
        )

        progress = speed * np.cos(angle)
        safe_speed = min(220.0, np.sqrt(max(front_dist, 1.0)) * 21.0)
        speed_ratio = speed / max(safe_speed, 1.0)

        progress_term = float(np.clip(progress / 34.0, -1.5, 4.2))
        speed_bonus = float(np.clip(speed / 95.0, 0.0, 2.2)) * float(
            np.clip(front_dist / 45.0, 0.2, 1.0)
        )
        straight_speed_bonus = straightness * float(np.clip(speed / 140.0, 0.0, 1.8))

        speed_gate = float(np.clip(speed / 30.0, 0.0, 1.0))
        center_term = speed_gate * 1.2 * (1.0 - min(track_pos, 1.0))
        heading_term = speed_gate * 1.2 * (1.0 - min(abs(angle) / 0.75, 1.0))

        steer_penalty = straightness * 1.2 * abs(steer)
        steer_delta_penalty = straightness * 3.0 * abs(steer - self.prev_steer)

        overspeed_penalty = 5.0 * (max(0.0, speed_ratio - 1.0) ** 2)
        edge_penalty = 4.0 * max(0.0, track_pos - 0.72)
        wall_penalty = 3.5 * max(0.0, (18.0 - front_dist) / 18.0)

        damage_delta = max(0.0, damage - self.prev_damage)
        self.prev_damage = damage
        damage_penalty = min(25.0, 0.06 * damage_delta)

        reward = (
            progress_term
            + speed_bonus
            + straight_speed_bonus
            + center_term
            + heading_term
            - steer_penalty
            - steer_delta_penalty
            - overspeed_penalty
            - edge_penalty
            - wall_penalty
            - damage_penalty
        )

        self.prev_steer = steer

        if progress < 3.0:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        force_done = False
        if self.stuck_steps > 80:
            reward -= 4.0
        if self.stuck_steps > 160:
            reward -= 12.0
            force_done = True

        if track_pos > 1.00:
            reward -= 12.0
            force_done = True

        if np.cos(angle) < -0.1:
            reward -= 10.0
            force_done = True

        if env_done:
            reward -= 8.0

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

        self.env.reset(relaunch=relaunch)
        self.prev_damage = float(self.env.client.S.d.get("damage", 0.0))
        return self._build_observation()

    def close(self):
        self.env.end()


def load_finetune_model(env, action_noise):
    speed_model_path = FINETUNE_MODEL_NAME + ".zip"
    speed_replay_path = FINETUNE_MODEL_NAME + "_replay_buffer.pkl"

    if os.path.exists(speed_model_path):
        print("Found existing speed model. Resuming speed fine-tuning...")
        model = TD3.load(speed_model_path, env=None, action_noise=action_noise)
        model.set_env(env)
        model.action_noise = action_noise
        if os.path.exists(speed_replay_path):
            try:
                model.load_replay_buffer(speed_replay_path)
                print("Loaded speed replay buffer.")
            except Exception as err:
                print("Could not load speed replay buffer:", err)
        return model

    for base_name in BASE_MODEL_CANDIDATES:
        base_model_path = base_name + ".zip"
        base_replay_path = base_name + "_replay_buffer.pkl"
        if os.path.exists(base_model_path):
            print("Loading base model for speed fine-tuning:", base_name)
            model = TD3.load(base_model_path, env=None, action_noise=action_noise)
            model.set_env(env)
            model.action_noise = action_noise
            if os.path.exists(base_replay_path):
                try:
                    model.load_replay_buffer(base_replay_path)
                    print("Loaded replay buffer from", base_name)
                except Exception as err:
                    print("Could not load replay buffer from", base_name, ":", err)
            return model

    print("No base model found. Expected one of:", ", ".join(BASE_MODEL_CANDIDATES))
    sys.exit(1)


if __name__ == "__main__":
    env = None
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.03 * np.ones(2))

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
