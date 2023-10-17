from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
import numpy as np
np.random.seed(50)

class Obstacle(Node):
    def __init__(self):
        super().__init__('obstacle',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
            
        # Get parameters from launch file
        self.obstacle_id = self.get_parameter('obstacle_id').value
        self.position = self.get_parameter('position').value
        self.max_iters = self.get_parameter('max_iters').value
        self.kk = 0

        # create the publisher
        self.publisher = self.create_publisher(
                                            MsgFloat, 
                                            f'/topic_obstacle_{self.obstacle_id}',
                                            10)
        timer_period = self.get_parameter('timer_period').value # [seconds]
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.kk > self.max_iters:
            print("\nMAXITERS reached")
            sleep(3) # [seconds]
            self.destroy_node()

        # Publish the updated message
        msg = MsgFloat()
        msg.data = [float(self.obstacle_id), float(self.kk), float(self.position[0]), float(self.position[1]), float(self.position[2])]
        self.publisher.publish(msg)
        if self.kk%20 == 0:
            self.get_logger().info(f"Obstacle {int(msg.data[0]):d} -- Iter: {int(msg.data[1]):d} \t position: {msg.data[2]:.4f},{msg.data[3]:.4f},{msg.data[4]:.4f}\n")
        # update iteration counter
        self.kk += 1

def main():
    rclpy.init()

    agent = Obstacle()
    agent.get_logger().info("GO!")

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()