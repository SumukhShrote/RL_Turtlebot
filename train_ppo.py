import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import os
import argparse
import rospy
from turtlebot3_gym_env import TurtleBot3Env  

parser = argparse.ArgumentParser(description='PyTorch PPO for TurtleBot3 controlling')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--env', type=str, default='TurtleBot3Env', help='continuous env')
parser.add_argument('--render', default=False, action='store_true', help='Render?')
parser.add_argument('--solved_reward', type=float, default=300, help='stop training if avg_reward > solved_reward')
parser.add_argument('--print_interval', type=int, default=5, help='how many episodes to print the results out')
parser.add_argument('--save_interval', type=int, default=50, help='how many episodes to save a checkpoint')
parser.add_argument('--max_episodes', type=int, default=10000)
parser.add_argument('--max_timesteps', type=int, default=150)
parser.add_argument('--update_timesteps', type=int, default=400, help='how many timesteps to update the policy')
parser.add_argument('--action_std', type=float, default=0.9, help='constant std for action distribution (Multivariate Normal)')
parser.add_argument('--K_epochs', type=int, default=80, help='update the policy for how long time everytime')
parser.add_argument('--eps_clip', type=float, default=0.3, help='epsilon for p/q clipped')
parser.add_argument('--gamma', type=float, default=0.75, help='discount factor')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=123, help='random seed to use')
parser.add_argument('--ckpt_folder', default='./checkpoints', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='./logs', help='Location to save logs')
parser.add_argument('--mode', default='train', help='choose train or test')
parser.add_argument('--restore', default=True, action='store_true', help='Restore and go on training?')
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:   # collected from old policy
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # Keeps output in the range [-1, 1]
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


        self.action_var = torch.full((action_dim, ), action_std * action_std).to(device)

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        state_value = self.critic(state)

        action_mean = self.actor(state).float()
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=False, ckpt=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # current policy
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        if restore:
            pretained_model = torch.load(ckpt, map_location=lambda storage, loc: storage)
            print(f"Loading model from checkpoint: {ckpt}")
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # old policy: initialize old policy with current policy's parameter
        self.old_policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()
    

    def select_action(self, state, info, memory):
        # Extract the numeric array and the goal position
        numeric_state = state  
        goal_position = info['goal_position']
        current_position = info['current_position']

        # Concatenate both parts into a single array
        state = np.concatenate([numeric_state.flatten(), goal_position.flatten()])
        state = np.concatenate([state, current_position.flatten()])

        # Convert the state to a FloatTensor and reshape for input
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        action = self.old_policy.act(state, memory)

        return action.detach().cpu().numpy().flatten()
        
    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(device, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(memory.states).to(device, dtype=torch.float32)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device, dtype=torch.float32)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device, dtype=torch.float32).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2)

            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

def find_latest_checkpoint(ckpt_folder):
        checkpoints = [f for f in os.listdir(ckpt_folder) if f.endswith('.pth')]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by episode number
        return os.path.join(ckpt_folder, checkpoints[-1])

def train(env, state_dim, action_dim, render, solved_reward, max_episodes, max_timesteps, update_timestep, action_std, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, restore, tb=False, print_interval=5, save_interval=50):

    memory = Memory()

    start_episode = 1
    checkpoint_path = find_latest_checkpoint(ckpt_folder) if restore else None

    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=True, ckpt=checkpoint_path)

    if checkpoint_path:
        # Parse the last saved episode number to resume from the correct episode
        start_episode = int(checkpoint_path.split('_')[-1].split('.')[0]) + 1

    running_reward, avg_length, time_step = 0, 0, 0

    for i_episode in range(start_episode, max_episodes+1):
        print("Episode:", i_episode)
        state, info, done = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            print("time_step: ", time_step)
            action = ppo.select_action(state, info, memory)
            state, reward, done, truncation, info = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += reward
            if render:
                env.render()

            if done:
                break
        avg_length += t

        if i_episode % save_interval == 0:
            torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_turtlebot_{}.pth'.format(str(i_episode)))
            print('Saved the model after {} episodes'.format(i_episode))

        if i_episode % print_interval == 0:
            avg_length = int(avg_length / print_interval)
            print("Running Reward: ", running_reward)
            print("Print Interval: ", print_interval)
            avg_reward = (running_reward / print_interval)
            print('Episode {} \t avg length: {} \t avg reward: {}'.format(i_episode, avg_length, avg_reward))
            running_reward, avg_length = 0, 0


if __name__ == '__main__':
    os.makedirs(opt.ckpt_folder, exist_ok=True)
    env = TurtleBot3Env()

    state_dim = env.observation_space.shape[0] + 4 #Adding 4 dimensions as current_position and target_postion are being added to state
    action_dim = env.action_space.shape[0]

    train(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        render=opt.render,
        solved_reward=opt.solved_reward,
        max_episodes=opt.max_episodes,
        max_timesteps=opt.max_timesteps,
        update_timestep=opt.update_timesteps,
        action_std=opt.action_std,
        K_epochs=opt.K_epochs,
        eps_clip=opt.eps_clip,
        gamma=opt.gamma,
        lr=opt.lr,
        betas=(0.9, 0.999),
        ckpt_folder=opt.ckpt_folder,
        restore=opt.restore,
        print_interval=opt.print_interval,
        save_interval=opt.save_interval
    )
