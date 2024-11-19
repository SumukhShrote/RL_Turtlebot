# RL-Based Autonomous Navigation with TurtleBot3

This project demonstrates a reinforcement learning (RL) approach for autonomous navigation of TurtleBot3 in a simulated environment using the Proximal Policy Optimization (PPO) algorithm. The goal of the agent is to reach a specific target while navigating through the environment. The state and goal positions are fed as inputs to the PPO agent, enabling precise navigation. The environment is built using the [TurtleBot3 Gym](https://github.com/ROBOTIS-GIT/turtlebot3_gazebo) and [ROS Noetic](https://www.ros.org/).

## Features
- **Goal-Reaching Task**: The agent is trained to navigate towards a goal, with the goal position refreshing after every episode.
- **Goal and State Input**: The model uses both the current position (state) and the goal position as inputs to the PPO model.
- **Simulated Environment**: The simulation uses the [TurtleBot3 Gazebo]((https://github.com/ROBOTIS-GIT/turtlebot3)) environment with ROS Noetic.
- **Reward Shaping**: Refined reward shaping to handle noisy data for stable training.
- **Training**: The agent is trained over approximately 5500 episodes.
