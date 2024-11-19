# RL-Based Autonomous Navigation with TurtleBot3

This project demonstrates a reinforcement learning (RL) approach for autonomous navigation of TurtleBot3 in a simulated environment using the Proximal Policy Optimization (PPO) algorithm. The goal of the agent is to reach a specific target while navigating through the environment. The state and goal positions are fed as inputs to the PPO agent, enabling precise navigation.

## Features
- **Goal-Reaching Task**: The agent is trained to navigate towards a goal, with the goal position refreshing after every episode.
- **Goal and State Input**: The model uses both the current position (state) and the goal position as inputs to the PPO model.
- **Simulated Environment**: The simulation uses the [TurtleBot3 Gazebo](https://github.com/ROBOTIS-GIT/turtlebot3) environment with ROS Noetic.
- **Reward Shaping**: Refined reward shaping to handle noisy data for stable training.
- **Training**: The agent is trained over approximately 5500 episodes.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- ROS Noetic
- TurtleBot3
- Gym
- Stable-Baselines3
- PyTorch
Setup

    Clone the repository:

git clone https://github.com/your-username/rl-turtlebot3-navigation.git
cd rl-turtlebot3-navigation

Add files to src folder: Move your custom environment files (gym_turtlebot3_navigation.py, train_ppo.py, checkpoint_test.py, etc.) to the src folder in the workspace.

Create and build the workspace: Navigate to your ROS workspace and build it using catkin_make.

cd ~/catkin_ws
catkin_make
source devel/setup.bash

Launch the TurtleBot3 simulation:

    roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

Training the Agent

To train the PPO agent, run the following script:

python src/train_ppo.py

This will start the training process in the TurtleBot3 simulation environment. The agent will learn to navigate toward the goal over the course of approximately 5500 episodes.
Goal Position Refresh

The goal position for each episode is refreshed once the agent either collides with an obstacle or reaches the goal. The environment will reset with a new goal for each episode.
Testing the Trained Model

Once training is complete, you can test a specific checkpoint by running:

python src/checkpoint_test.py --checkpoint <path_to_checkpoint>

This script will load the model checkpoint and evaluate the agent's performance on the goal-reaching task.
GIF Demonstration

Here's a demonstration of the agent's performance as it navigates towards the goal:

Conclusion

This project illustrates the application of reinforcement learning in autonomous navigation using the PPO algorithm. The model can navigate towards a goal in a dynamic and noisy environment and is ready for further improvements in robustness and generalization.
License

This project is licensed under the MIT License - see the LICENSE file for details.


### Additional Notes:
- **Add Your GIF**: Replace `path_to_your_gif.gif` with the actual path to your GIF demonstrating the model's performance.
- **Repository URL**: Replace `https://github.com/your-username/rl-turtlebot3-navigation.git` with the actual URL of your GitHub repository.
- **ROS Launch File**: If you have any specific launch file setups or other configurations, be sure to adjust the launch command in the instructions.

This `README.md` should provide clear, structured guidance for setting up, training, and te
