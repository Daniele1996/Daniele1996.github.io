#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

class ServiceCaller(Node):

    def __init__(self):
        super().__init__('ServiceCaller')

        self.client = self.create_client(Empty, '/reinitialize_global_localization')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available')
        self.req = Empty.Request()
        
    def call_service(self):
        future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)


def main(args=None):

    rclpy.init()
    node  = ServiceCaller()
    node.call_service()
    rclpy.shutdown()

if __name__ == '__main__':
    main()