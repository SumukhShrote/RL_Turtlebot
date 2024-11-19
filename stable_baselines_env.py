import gymnasium as gym
from gymnasium import spaces
import rospy
import subprocess
import os
import signal
import numpy as np
import random
import sys
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments."""
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile):
        self.last_clock_msg = Clock()

        #random_number = random.randint(10000, 15000)
        self.port = str(11311) 
        #self.port_gazebo = str(random_number + 1)

        # Set the TurtleBot3 model environment variable
        os.environ["TURTLEBOT3_MODEL"] = "waffle"

        os.environ["ROS_MASTER_URI"] = "http://localhost:" + self.port
        #os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + self.port_gazebo

        print("TURTLEBOT3_MODEL=waffle")
        print("ROS_MASTER_URI=http://localhost:" + self.port + "\n")
        #print("GAZEBO_MASTER_URI=http://localhost:" + self.port_gazebo + "\n")

        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"])).decode('utf-8')

        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", "launch", launchfile)
        if not os.path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        self._roslaunch = subprocess.Popen([sys.executable, os.path.join(ros_path, "roslaunch"), "-p", self.port, fullpath])
        print("Gazebo launched!")

        self.gzclient_pid = 0

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True)

        # Wait for the ROS master to be ready
        while not rospy.is_shutdown():
            try:
                rospy.get_master().getSystemState()
                break
            except rospy.ROSException:
                time.sleep(1)  # Wait a second before retrying

    def step(self, action):
        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):
        # Implemented in subclass
        raise NotImplementedError

    def _render(self, mode="human", close=False):
        if close:
            if self.gzclient_pid != 0:
                os.kill(self.gzclient_pid, signal.SIGTERM)
                os.wait()
            return

        if self.gzclient_pid == 0:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof", "-s", "gzclient"]))

    def _close(self):
        # Kill gzclient, gzserver and roscore
        os.system("killall -9 gzclient gzserver roscore rosmaster")

    def _configure(self):
        # TODO: Provides runtime configuration to the environment
        pass

    def _seed(self):
        # TODO: Sets the seed for this env's random number generator(s)
        pass


class TurtleBot3Env(GazeboEnv):

    def __init__(self):
        super().__init__("/opt/ros/noetic/share/turtlebot3_gazebo/launch/turtlebot3_world.launch")
        
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        #self.sub_laser = rospy.Subscriber('/scan', LaserScan, self._laser_callback)
        #self.sub_odom = rospy.Subscriber('/odom', Odometry, self._odom_callback)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Define action space (linear and angular velocity)
        self.action_space = spaces.Box(low=np.array([-0.2, -1.0]), high=np.array([0.2, 1.0]), dtype=np.float32)
        
        # Define observation space (Laser Scan data - 360 degrees)
        self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(3.5), shape=(364,), dtype=np.float32)
        
        # Goal position (x, y) in the environment
        self.goal_position = np.array([0.0, 0.0])  # Modify as needed

        self.done = False
        self.prev_distance_to_goal = float("inf")

    # def _laser_callback(self, data):
    #     self.laser_data = data

    # def _odom_callback(self, data):
    #     self.odom_data = data

    def _seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action):
        self.done = False
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # Publish Twist command for linear and angular velocity
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)

        # Wait for sensor data
        rospy.sleep(0.1)

        # Laser Scan Data
        laser_data = None
        while laser_data is None:
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                #print("Laser data ranges:", laser_data.ranges)  # Debugging laser data
            except Exception as e:
                print(f"Error receiving laser data: {e}")

        # Current Position
        odom_data = None
        while odom_data is None:
            try:
                odom_data = rospy.wait_for_message('/odom', Odometry, timeout=5)
            except:
                pass

        current_position = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y], dtype=np.float32)

        # Update observation
        observation = np.append(laser_data.ranges, current_position)
        observation = np.append(observation, self.goal_position.astype(np.float32))

        # Ensure valid observations
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        observation = np.nan_to_num(observation, nan=3.5, posinf=3.5, neginf=3.5).astype(np.float32)

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(self.goal_position - current_position)
        print("Distance_to_goal: ", distance_to_goal)
        # Reward system
        reward = 0.0
        min_range = min(observation[:360])  # Only consider laser scan ranges
        print("Min Range: ", min_range)
        
        if min_range < 0.2:
            reward -= 100.0  # Collision penalty
            self.done = True
            print("Collision occurred!")
        else:
            self.done = False

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # Reward for progress
        if distance_to_goal < self.prev_distance_to_goal:
            reward += 1
        else:
            reward += -0.5 * distance_to_goal
        
        self.prev_distance_to_goal = distance_to_goal

        # Goal reached
        if distance_to_goal < 0.5:
            print("Goal Reached !!!")
            reward += 100.0
            self.done = True

        return observation, reward, self.done, False, {}

    def reset(self, seed=None):
        if seed is not None:
            self._seed(seed)

        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        laser_data = None
        while laser_data is None:
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        
        odom_data = None
        while odom_data is None:
            try:
                odom_data = rospy.wait_for_message('/odom', Odometry, timeout=5)
            except:
                pass
        
        # Reset goal to a new random position (optional)
        #self.goal_position = np.random.uniform(low=0.0, high=10.0, size=(2,))

        observation = np.array(laser_data.ranges, dtype=np.float32)
        
        current_position = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y], dtype=np.float32)
        observation = np.append(observation, current_position)
        observation = np.append(observation, self.goal_position.astype(np.float32))

        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        observation = np.nan_to_num(observation, nan=0.0, posinf=3.5, neginf=0.0).astype(np.float32)


        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        info = {}  
        #self.done = False
        return observation, info#, self.done
