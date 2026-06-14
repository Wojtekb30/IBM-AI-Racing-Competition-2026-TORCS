import os
import gym
import numpy as np
from gym import spaces
from gym_torcs import TorcsEnv
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit


class TorcsFlatEnv(gym.Env):
    def __init__(self, vision=False, throttle=True, action_repeat=2):
        super(TorcsFlatEnv, self).__init__()
        self.env = TorcsEnv(vision=vision, throttle=throttle, gear_change=False)
        self.action_space = self.env.action_space
        self.action_repeat = action_repeat

        self.env.terminal_judge_start = 4000
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

    def _as_vector(self, value, size):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= size:
            return arr[:size]
        out = np.zeros(size, dtype=np.float32)
        out[:arr.size] = arr
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
        act = np.asarray(action, dtype=np.float32).copy()
        steer = float(np.clip(act[0], -1.0, 1.0))
        throttle = float(np.clip((act[1] + 1.0) / 2.0, 0.0, 1.0))

        raw_obs = self.env.client.S.d
        track = self._as_vector(raw_obs.get("track", []), 19)
        front_dist = float(np.max(track[8:11]))
        abs_angle = abs(float(raw_obs.get("angle", 0.0)))
        speed = float(raw_obs.get("speedX", 0.0))

        danger = min(1.0, (abs_angle / 0.70) + max(0.0, (30.0 - front_dist) / 30.0))
        throttle_cap = max(0.20, 1.0 - (0.70 * danger))
        throttle = min(throttle, throttle_cap)

        if speed < 5.0 and front_dist > 20.0:
            throttle = max(throttle, 0.35)

        return np.array([steer, throttle], dtype=np.float32)

    def _shape_reward(self, env_done):
        raw_obs = self.env.client.S.d
        track = self._as_vector(raw_obs.get("track", []), 19)

        speed = float(raw_obs.get("speedX", 0.0))
        angle = float(raw_obs.get("angle", 0.0))
        track_pos = abs(float(raw_obs.get("trackPos", 0.0)))
        front_dist = float(np.max(track[8:11]))
        damage = float(raw_obs.get("damage", 0.0))

        progress = speed * np.cos(angle)
        progress_term = float(np.clip(progress / 35.0, -1.0, 3.0))

        speed_gate = float(np.clip(speed / 30.0, 0.0, 1.0))
        center_term = speed_gate * 1.5 * (1.0 - min(track_pos, 1.0))
        heading_term = speed_gate * 1.0 * (1.0 - min(abs(angle) / 0.8, 1.0))

        edge_penalty = 2.5 * max(0.0, track_pos - 0.85)
        wall_penalty = 2.5 * max(0.0, (20.0 - front_dist) / 20.0)

        damage_delta = max(0.0, damage - self.prev_damage)
        self.prev_damage = damage
        damage_penalty = min(12.0, 0.03 * damage_delta)

        reward = (
            progress_term
            + center_term
            + heading_term
            - edge_penalty
            - wall_penalty
            - damage_penalty
        )

        if progress < 2.0:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        force_done = False
        if self.stuck_steps > 60:
            reward -= 3.0
        if self.stuck_steps > 140:
            reward -= 12.0
            force_done = True

        if np.cos(angle) < -0.2:
            reward -= 8.0
            force_done = True

        if env_done:
            reward -= 2.0

        return float(np.clip(reward, -20.0, 20.0)), force_done

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        safe_action = self._safe_action(action)

        for _ in range(self.action_repeat):
            _, _, done, info = self.env.step(safe_action)
            shaped_reward, force_done = self._shape_reward(done)
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

        self.env.reset(relaunch=relaunch)
        self.prev_damage = float(self.env.client.S.d.get("damage", 0.0))
        return self._build_observation()

    def close(self):
        self.env.end()


def build_td3_model(env, action_noise):
    return TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=300000,
        learning_starts=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        train_freq=(4, "step"),
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
    )


if __name__ == "__main__":
    learn_chunk_timesteps = 15000
    save_name = "td3_torcs_trackaware_v1"
    model_path = save_name + ".zip"
    replay_buffer_path = save_name + "_replay_buffer.pkl"
    env = None

    print("Preparing TD3 model...")
    n_actions = 2
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.08 * np.ones(n_actions))

    if os.path.exists(model_path):
        print("Found existing model. Resuming training...")
        try:
            model = TD3.load(model_path, env=None, action_noise=action_noise)
            if os.path.exists(replay_buffer_path):
                model.load_replay_buffer(replay_buffer_path)
                print("Replay buffer loaded.")
            env = TorcsFlatEnv(vision=False, throttle=True, action_repeat=2)
            model.set_env(env)
            model.action_noise = action_noise
        except Exception as err:
            print("Model load failed:", err)
            print("Starting TD3 from scratch instead.")
            env = TorcsFlatEnv(vision=False, throttle=True, action_repeat=2)
            model = build_td3_model(env, action_noise)
    else:
        print("No previous model found. Training TD3 from scratch...")
        env = TorcsFlatEnv(vision=False, throttle=True, action_repeat=2)
        model = build_td3_model(env, action_noise)

    model.train_freq = TrainFreq(4, TrainFrequencyUnit.STEP)
    model.gradient_steps = 1

    print("Starting continuous TD3 training (Ctrl+C to stop and save)...")
    chunks_completed = 0
    try:
        while True:
            try:
                model.learn(
                    total_timesteps=learn_chunk_timesteps,
                    log_interval=10,
                    reset_num_timesteps=False,
                )
                chunks_completed += 1
                print("Completed chunks:", chunks_completed)
                if chunks_completed % 2 == 0:
                    model.save(save_name)
                    model.save_replay_buffer(replay_buffer_path)
                    print("Periodic save complete.")
            except Exception as err:
                print("Learn chunk failed:", err)
                print("Rebuilding TORCS environment and continuing...")
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                env = TorcsFlatEnv(vision=False, throttle=True, action_repeat=2)
                model.set_env(env)
                model.action_noise = action_noise
                model.train_freq = TrainFreq(4, TrainFrequencyUnit.STEP)
                model.gradient_steps = 1
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        if env is not None:
            env.close()
        model.save(save_name)
        model.save_replay_buffer(replay_buffer_path)
        print("Model and replay buffer saved.")
