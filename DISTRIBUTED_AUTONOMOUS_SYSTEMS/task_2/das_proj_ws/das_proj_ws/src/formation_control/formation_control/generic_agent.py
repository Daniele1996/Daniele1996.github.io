from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
import numpy as np
np.random.seed(50)

class Agent(Node):
    def __init__(self):
        super().__init__('agent',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
            
        # Get parameters from launch file
        self.agent_id = self.get_parameter('agent_id').value
        self.neigh = self.get_parameter('neigh').value
        self.all_agents = self.get_parameter('all_agents').value
        self.Adj_row = self.get_parameter('Adj_row').value # row of the Adjacency matrix
        self.distances = self.get_parameter('distances_row').value # row of the distances matrix
        self.planar_formation = self.get_parameter('planar_formation').value # wether the formation needs to be obtained on a plane or not
        self.x_i = np.asarray(self.get_parameter('x_init').value, dtype=np.float32) # 3D state vector [x,y,z] 
        self.kk = 0 # iteration counter
        self.max_iters = self.get_parameter('max_iters').value
        self.euler_step = self.get_parameter('euler_step').value
        self.agent_in_position = 0 # wether the agent is in position (1) or not (0)
        self.agent_goal = self.get_parameter('goal').value # position to reach 
        self.agent_speed = self.get_parameter('agent_speed').value
        self.moving = False # wether the agent is moving (1) or not (0)
        self.N_obstacle = self.get_parameter('N_obstacle').value

        # create a subscription to each other agent
        for j in self.all_agents:
            self.create_subscription(
                                    MsgFloat, 
                                    f'/topic_agent_{j}', #  topic_name
                                    self.listener_callback, 
                                    10)
            
        # create a subscription to each obstacle
        for obst in range(self.N_obstacle):
            self.create_subscription(
                                    MsgFloat, 
                                    f'/topic_obstacle_{obst}', #  topic_name
                                    self.listener_callback_obst, 
                                    10)
            
        # create the publisher
        self.publisher = self.create_publisher(
                                    MsgFloat, 
                                    f'/topic_agent_{self.agent_id}',
                                    10)
        
        timer_period = self.get_parameter('timer_period').value # [seconds]
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # initialize a dictionary with a list of received messages from each neighbor j [a queue]
        self.received_data = { j:[] for j in self.all_agents }
        # self.received_data = { j:[] for j in self.neigh }
        # initialize a dictionary with a list of received messages from each obstacle j [a queue]
        self.obstacle_pos = [ [] for _ in range(self.N_obstacle) ]

        print(f"Setup of agent {self.agent_id} completed")

    def listener_callback(self, msg):
        self.received_data[int(msg.data[0])].append(list(msg.data[1:])) 
    
    def listener_callback_obst(self, msg):
        self.obstacle_pos[int(msg.data[0])]= msg.data[2:]

    def timer_callback(self):
        # Skip the first iteration
        all_received = False
        gain_collision = 1 # gain of the collision avoidance potential
        gain_formation = 6 # gain of the formation potential
        gain_obstacle = 4 # gain of the obstacle avoidance potential
        gain_planar = 3 # gain of the elastic potential introduced for the planar formations
        if self.kk > 0:
            # Have all messages at time kk-1 been received? Check top message in each queue
            try:
                all_received = all(self.kk-1 == self.received_data[j][0][0] for j in self.all_agents) # True if all True
                # all_received = all(self.kk-1 == self.received_data[j][0][0] for j in self.neigh) # True if all True
                # # Checking if the agent and its neighbours are in formation 
                all_in_position = all(self.received_data[j][0][-1] and self.agent_in_position for j in self.neigh)
            except:
                if self.kk%20==0:
                    self.get_logger().info(f'Waiting for others agents to receive data...')

            if all_received:
                
                # initialize DV_sum for each agent
                DV_sum = np.zeros(3, dtype=np.float32)
                # initialize the list of positions
                x_vector = []
                # save positions of each agent
                for agent in range(len(self.Adj_row)):
                    if agent == self.agent_id:
                        x_vector.append(self.x_i)
                        continue
                    # extract x_j from the j-th buffer queue
                    _, x_j1, x_j2, x_j3, _ = self.received_data[agent].pop(0)
                    # create the 3D position vector of each neighbour
                    x_j = np.array([x_j1,x_j2,x_j3], dtype=np.float32)
                    x_vector.append(x_j)
                    # add the contribution of the collision potential
                    if np.linalg.norm(self.x_i - x_j) < 0.7:
                        DV_sum += gain_collision*(self.x_i-x_j)/np.linalg.norm(self.x_i-x_j)**2

                #  add the contribution of the formation potential
                if not self.moving:
                    for j in self.neigh:
                        DV_sum -= gain_formation*(np.linalg.norm(self.x_i-x_vector[j])**2 - self.distances[j]**2)*(self.x_i - x_vector[j])
                else:
                    if self.agent_id != 0:
                        for j in self.neigh:
                            DV_sum -= gain_formation*(np.linalg.norm(self.x_i-x_vector[j])**2 - self.distances[j]**2)*(self.x_i - x_vector[j])
                
                # add the contribution of the obstacle avoidance
                for obst in self.obstacle_pos:
                    obst_distance = np.linalg.norm(self.x_i-np.array(obst,dtype=np.float32))
                    DV_sum += gain_obstacle*(self.x_i-np.array(obst,dtype=np.float32))/obst_distance**5 # hyperbolic bar func

                # if necessary add the contribution of the planar potential
                if self.planar_formation:
                    DV_sum -= gain_planar*np.array([0,0,self.x_i[2]]) # in case we are interested in mantaining a given distance from a plane in this case z=0
                
                # check when the agent is in the right position
                if np.linalg.norm(DV_sum) < 0.1:
                    self.agent_in_position = 1
                    if self.kk%20==0:
                        self.get_logger().info(f"!Agent: {self.agent_id} in position! waiting for the others..\n")

                # if the formation is completed then let the first agent move
                if all_in_position or self.moving:
                    self.moving = True
                    if self.agent_id==0:
                        position_error = np.array(self.agent_goal,dtype=np.float32)-self.x_i
                        self.x_i += self.euler_step*self.agent_speed*position_error # update the position of the agent who's moving
                        if self.kk%20==0:
                            self.get_logger().info(f"!Agent: {self.agent_id} moving with velocity Vx: {self.euler_step*self.agent_speed*position_error}!\n")
                    else:
                        if self.kk%20==0:
                            self.get_logger().info(f"!Agent: {self.agent_id} trying to keep the formation intact!\n")
                
                # update the local state
                self.x_i +=  self.euler_step * DV_sum # Euler integration step
                
                # Stop the node if kk exceeds the maximum iteration
                if self.kk%20==0:
                    self.get_logger().info(f"Norm of DV_sum for agent {self.agent_id}: {np.linalg.norm(DV_sum)}\n")
                if self.kk > self.max_iters:
                    print("\nMAXITERS reached")
                    sleep(3) # [seconds]
                    self.destroy_node()
                
                msg = MsgFloat()
                msg.data = [float(self.agent_id), float(self.kk), float(self.x_i[0]), float(self.x_i[1]), float(self.x_i[2]), float(self.agent_in_position)]
                self.publisher.publish(msg)

                if self.kk%20==0:
                    self.get_logger().info(f"Agent {int(msg.data[0]):d} -- Iter: {int(msg.data[1]):d} \t Value: {msg.data[2]:.4f},{msg.data[3]:.4f},{msg.data[4]:.4f} \t Agent_in_position: {msg.data[5]}\n")
                # update iteration counter
                self.kk += 1
        else: 
            # Publish the updated message
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.kk), float(self.x_i[0]), float(self.x_i[1]), float(self.x_i[2]), float(self.agent_in_position)]
            self.publisher.publish(msg)

            self.get_logger().info(f"Agent {int(msg.data[0]):d} -- Iter: {int(msg.data[1]):d} \t Value: {msg.data[2]:.4f},{msg.data[3]:.4f},{msg.data[4]:.4f} \t Agent_in_position: {msg.data[5]}\n")
            # update iteration counter
            self.kk += 1

def main():
    rclpy.init()

    agent = Agent()
    agent.get_logger().info(f"Agent {agent.agent_id:d} -- Waiting for sync...")
    sleep(1)
    agent.get_logger().info("GO!")

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()