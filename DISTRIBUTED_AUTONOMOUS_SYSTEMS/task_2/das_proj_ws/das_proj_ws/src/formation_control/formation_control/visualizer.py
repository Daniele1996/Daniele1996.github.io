import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray as msg_float

class Visualizer(Node):

    def __init__(self):
        super().__init__('visualizer',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
        
        # Get parameters from launcher
        self.agent_id = self.get_parameter('agent_id').value
        self.obstacle_id = self.get_parameter('obstacle_id').value
        self.agent_or_not = self.get_parameter('agent_or_not').value

        self.node_frequency = self.get_parameter('node_frequency').value


        #######################################################################################
        # Let's subscribe to the topic we want to visualize
        if self.agent_or_not:
            self.subscription = self.create_subscription(
                                                        msg_float, 
                                                        '/topic_agent_{}'.format(self.agent_id),
                                                        self.listener_callback, 
                                                        10)
        else:
            self.subscription = self.create_subscription(
                                                        msg_float, 
                                                        '/topic_obstacle_{}'.format(self.obstacle_id),
                                                        self.listener_callback, 
                                                        10)

        #######################################################################################

        # Create the publisher that will communicate with Rviz
        self.timer = self.create_timer(1.0/self.node_frequency, self.publish_data)
        self.publisher = self.create_publisher(
                                                Marker, 
                                                '/visualization_topic', 
                                                1)

        # Initialize the current_pose method (in this example you can also use list or np.array)                                         
        self.current_pose = Pose()


    def listener_callback(self, msg):
        # store (and rearrange) the received message
        self.current_pose.position.x = msg.data[2] # fix x coordinate
        self.current_pose.position.y = msg.data[3]
        self.current_pose.position.z = msg.data[4]
            
    def publish_data(self):
        if self.current_pose.position is not None:
            # Set the type of message to send to Rviz -> Marker
            # (see http://docs.ros.org/en/noetic/api/visualization_msgs/html/index-msg.html)
            marker = Marker()

            # Select the name of the reference frame, without it markers will be not visualized
            marker.header.frame_id = 'my_frame'
            marker.header.stamp = self.get_clock().now().to_msg()

            # Select the type of marker
            if self.agent_or_not:
                marker.type = Marker.SPHERE
            else:
                marker.type = Marker.CUBE

            # set the pose of the marker (orientation is omitted in this example)
            marker.pose.position.x = self.current_pose.position.x
            marker.pose.position.y = self.current_pose.position.y
            marker.pose.position.z = self.current_pose.position.z 

            # Select the marker action (ADD, DELATE)
            marker.action = Marker.ADD

            # Select the namespace of the marker
            if self.agent_or_not:
                marker.ns = 'agents'
                # Let the marker be unique by setting its id
                marker.id = self.agent_id
            else:
                marker.ns = 'obstacles'
                marker.id = self.obstacle_id

            # Specify the scale of the marker
            if self.agent_or_not:
                scale = 0.2
            else:
                scale = 0.8

            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Specify the color of the marker as RGBA
            color = [1.0, 0.0, 0.0, 1.0]
            # if self.agent_id % 2:
            if self.agent_or_not:
                if self.agent_id >0:
                    color = [0.0, 0.0, 1.0-self.agent_id*0.1, 1.0]
            else:
                color = [0.0, 0.0, 0.0, 1.0]

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            # Let's publish the marker
            self.publisher.publish(marker)

def main():
    rclpy.init()

    visualizer = Visualizer()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("----- Visualizer stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()