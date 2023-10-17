#!/usr/bin/env python3

import os
import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import FollowWaypoints
import tf_transformations
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.action import ActionClient
import numpy as np
import yaml
from std_msgs.msg import Bool
from ament_index_python.packages import get_package_share_directory


#this function creates poses

def create_pose_stamped(position_x, position_y, orientation_z):   

    q_x, q_y, q_z, q_w = tf_transformations.quaternion_from_euler(0.0, 0.0, orientation_z)
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = position_x
    pose.pose.position.y = position_y
    pose.pose.position.z = 0.0
    pose.pose.orientation.x = q_x
    pose.pose.orientation.y = q_y
    pose.pose.orientation.z = q_z
    pose.pose.orientation.w = q_w
    return pose

class Navigator(Node):

    def __init__(self):
        super().__init__('Navigator')

        self.follow_waypoints_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        self.amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)
    
        self.localization_ended = False
        self.estim_amcl_pose = PoseStamped()
        self.estim_amcl_orientation = PoseStamped()
        self.estim_amcl_cov = np.empty((6,6))

        self.amcl_pose = self.create_subscription(
                    PoseWithCovarianceStamped,
                    'amcl_pose',
                    self._amclPoseCallback,
                    self.amcl_pose_qos
                    )

        self.loc_subscriber = self.create_subscription(
                    Bool,
                    'LocalizationCompleted',
                    self.localization_callback,
                    10
                    )
        
        self.call_timer = self.create_timer(5, self.timer_cb)

        mod_dir = os.path.join(get_package_share_directory('task3'), 'config', 'task3_nav_goals.yaml')
        with open(mod_dir , 'r') as file:
            yaml_file = file.read()
        nav_goals_yaml = yaml.safe_load(yaml_file)
        self.nav_goals = np.zeros((len(nav_goals_yaml),3))
        for i in range(len(nav_goals_yaml)):
            self.nav_goals[i,0] = nav_goals_yaml['goal_'+str(i)]['pose']['position']['x']
            self.nav_goals[i,1] = nav_goals_yaml['goal_'+str(i)]['pose']['position']['y']
            self.nav_goals[i,2] = nav_goals_yaml['goal_'+str(i)]['pose']['orientation']['z']
        self.get_logger().info("Loaded nav_goals.yaml file %s\n" % self.nav_goals)
    
    def _amclPoseCallback(self, msg):
        self.initial_pose_received = True
        self.estim_amcl_pose = msg.pose.pose.position
        self.estim_amcl_orientation = msg.pose.pose.orientation
        self.estim_amcl_cov = np.reshape(msg.pose.covariance, (6,6))
        self.get_logger().info('Check on Eigenvalues: "%s"' % np.linalg.eigvals(self.estim_amcl_cov))
    
    def localization_callback(self,msg):
        self.localization_ended = msg.data

    def followWaypoints(self, poses):
        """Send a `FollowWaypoints` action request."""
        self.get_logger().debug("Waiting for 'FollowWaypoints' action server")
        while not self.follow_waypoints_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("'FollowWaypoints' action server not available, waiting...")

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = poses

        self.get_logger().info(f'Following {len(goal_msg.poses)} goals....')
        send_goal_future = self.follow_waypoints_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.get_logger().error(f'Following {len(poses)} waypoints request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def timer_cb(self):
            
            if not self.localization_ended:
                self.get_logger().info('waiting for localization to end...')
                return
            
            else: 
                self.get_logger().info('Starting navigation with initial pose: x: "%f", y :"%f, theta: "%f"' % (self.estim_amcl_pose.x, self.estim_amcl_pose.y, tf_transformations.euler_from_quaternion([self.estim_amcl_orientation.x ,self.estim_amcl_orientation.y, self.estim_amcl_orientation.z,self.estim_amcl_orientation.w])[2]))

                # --- Send Nav2 goal
                waypoints = []
                for j in range(self.nav_goals.shape[0]):
                    waypoints.append(create_pose_stamped(self.nav_goals[j,0],self.nav_goals[j,1],self.nav_goals[j,2]))

                # --- Follow waypoints
                self.get_logger().info("Navigating to: %s" % (self.nav_goals))
                self.followWaypoints(waypoints)
                self.get_logger().info("Navigation Completed!!!")

                rclpy.shutdown()

def main():
    # --- Init
    rclpy.init()

    try:
        navigator = Navigator()
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        # --- Shutdown
        rclpy.shutdown()

if __name__ == '__main__':
    main()
