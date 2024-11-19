import torch
import torch.nn as nn
import numpy as np
from turtlebot3_gym_env import TurtleBot3Env
from argparse import Namespace

env = TurtleBot3Env()

# Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # Actions should be between -1 and 1
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output for value function
        )

        self.action_var = torch.full((action_dim,), action_std ** 2).to(device)

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        return action

    def evaluate(self, state):
        value = self.critic(state)
        return value


opt = Namespace(
    checkpoint_path='./checkpoints/PPO_continuous_turtlebot_5300.pth',
    action_std=0.9,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


state_dim = env.observation_space.shape[0] + 4  # Adding dimensions for goal and current positions
action_dim = env.action_space.shape[0]


def evaluate_checkpoint(checkpoint_path, num_episodes=5):
    print(f"Evaluating checkpoint: {checkpoint_path}")
    policy = ActorCritic(state_dim, action_dim, opt.action_std).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()

    total_rewards = []

    for episode in range(num_episodes):
        state, info, done = env.reset()
        episode_reward = 0

        while not done:
            numeric_state = state
            goal_position = info['goal_position']
            current_position = info['current_position']

            # Combine state, goal position, and current position
            combined_state = np.concatenate([
                numeric_state.flatten(), goal_position.flatten(), current_position.flatten()
            ])
            state_tensor = torch.FloatTensor(combined_state).unsqueeze(0).to(device)

            # Get action from the pretrained model
            action = policy.act(state_tensor).detach().cpu().numpy().flatten()

            # Take action in the environment
            state, reward, done, truncation, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Checkpoint: {checkpoint_path}, Episode: {episode + 1}, Reward: {episode_reward}")

    average_reward = np.mean(total_rewards)
    print(f"Checkpoint: {checkpoint_path}, Average Reward: {average_reward}\n")
    return average_reward

average_reward = evaluate_checkpoint(opt.checkpoint_path)
print(f"Checkpoint {opt.checkpoint_path} achieved an average reward of {average_reward}")

env._close()
