import numpy as np
import gym
from gym import spaces
from gym_torcs import TorcsEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import math
import os
import time

# ================= EXPERT DRIVER LOGIC =================
def get_expert_action(S):
    """Replicates the logic from torcs_jm_par.py to output [steer, accel]"""
    # 1. Steering Logic
    steer = (S['angle'] * 50 / math.pi) - (S['trackPos'] * 0.60)
    steer = max(-1.0, min(1.0, steer))
    
    # 2. Throttle Logic
    TARGET_SPEED = 180  # Keep expert slightly conservative so DDPG can learn to go faster
    if 'track' in S and len(S['track']) == 19:
        front_dist = max(S['track'][8], S['track'][9], S['track'][10])
    else:
        front_dist = 200.0
    
    front_dist = max(front_dist, 1)
    safe_speed = math.sqrt(front_dist) * 16.5
    target_speed = min(TARGET_SPEED, safe_speed)

    if S['speedX'] > target_speed:
        accel = 0.0
    else:
        accel = 0.8
        if S['speedX'] < 10:
            accel = 1.0
        # Traction control
        if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) - 
            (S['wheelSpinVel'][0] + S['wheelSpinVel'][1])) > 2:
            accel -= 0.2
            
    accel = max(0.0, min(1.0, accel))
    return np.array([steer, accel], dtype=np.float32)

# ================= ENVIRONMENT WRAPPER =================
class TorcsFlatEnv(gym.Env):
    def __init__(self):
        super(TorcsFlatEnv, self).__init__()
        self.env = TorcsEnv(vision=False, throttle=True, gear_change=False)
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float32)
        self.reset_count = 0

    def _flatten_obs(self, obs):
        flat = np.concatenate([
            np.array([obs.speedX, obs.speedY, obs.speedZ, obs.rpm]),
            obs.track, obs.wheelSpinVel, obs.opponents, obs.focus
        ])
        return flat.astype(np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._flatten_obs(obs), reward, done, info

    def reset(self):
        self.reset_count += 1
        relaunch = (self.reset_count % 3 == 0)
        obs = self.env.reset(relaunch=relaunch)
        return self._flatten_obs(obs)
        
    def close(self):
        self.env.end()

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    print("1. Initializing Environment and DDPG Model...")
    env = TorcsFlatEnv()
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))

    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise, 
        verbose=1,
        buffer_size=50000,
        learning_starts=9999999, # PREVENT DDPG FROM TRAINING AUTOMATICALLY YET
        batch_size=64,
        gamma=0.99
    )

    vec_env = model.get_env()
    obs = vec_env.reset()

    # --- PHASE 1: COLLECT EXPERT DATA ---
    expert_steps = 3000 # Roughly 1 full lap on most tracks
    print(f"\n2. Collecting {expert_steps} steps from the Expert Driver...")
    print("   (PyTorch is idle, so this should run smoothly without timeouts)")
    
    for step in range(expert_steps):
        # Dig into the wrapper to get the raw TORCS dictionary for the expert logic
        raw_obs = vec_env.envs[0].env.client.S.d
        
        # Expert takes control
        action = np.array([get_expert_action(raw_obs)]) 
        
        # Step the environment
        next_obs, rewards, dones, infos = vec_env.step(action)
        
        # Force inject the expert's success into DDPG's brain (Replay Buffer)
        model.replay_buffer.add(obs, next_obs, action, rewards, dones, infos)
        obs = next_obs
        
        if step % 500 == 0:
            print(f"   ...Collected {step}/{expert_steps} expert transitions")

    # --- PHASE 2: OFFLINE TRAINING (DOCKER-SAFE) ---
    print("\n3. Closing TORCS for Offline Training...")
    os.system('pkill torcs') # Kill the game so it doesn't timeout!
    
    print("4. Training Neural Networks on Expert Data (This may take a minute...)")
    # We force the model to do 2000 math updates. 
    # Because TORCS is closed, your Docker CPU can take as long as it needs!
    model.train(gradient_steps=2000, batch_size=64)
    
    print("   Saving pre-trained Expert Model...")
    model.save("ddpg_expert_pretrained")

    # --- PHASE 3: RL FINE-TUNING ---
    print("\n5. Relaunching TORCS for RL Fine-Tuning...")
    # Re-enable standard learning
    model.learning_starts = 0 
    model.train_freq = (4, "step") # Train 1 time every 4 steps to save CPU
    model.gradient_steps = 1
    
    # Force a hard reset to get TORCS back up and running safely
    vec_env.envs[0].reset_count = 2 
    obs = vec_env.reset()

    print("6. AI is now taking over to optimize speed! Watch it drive.")
    try:
        model.learn(total_timesteps=50000, log_interval=10, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        model.save("ddpg_torcs_finetuned")
        env.close()
        print("Final Model Saved.")