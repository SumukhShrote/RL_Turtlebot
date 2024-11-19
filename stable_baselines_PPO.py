import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines_env import TurtleBot3Env

def make_env():
    return TurtleBot3Env()

vec_env = DummyVecEnv([make_env])

# Instantiate the PPO model with the specified policy and environment
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the model for the specified number of timesteps
model.learn(total_timesteps=25000)

# Save the trained model
model.save("ppo_turtlebot3")

del model 

model = PPO.load("ppo_turtlebot3")

# Reset the environment
obs, _ = vec_env.reset()

while True:

    action, _states = model.predict(obs)

    # Step the environment with the predicted action
    obs, rewards, dones, truncations, info = vec_env.step(action)

    # Check if the episode is done or truncated
    if dones or truncations:
        obs, info = vec_env.reset()

    # Render the environment
    vec_env.render()
