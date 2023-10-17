from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os
import numpy as np
import networkx as nx
from pprint import pprint 
np.random.seed(50)

# Usefull Parameters
N = 6 # Number of Agents
n_x = 3 # state vector dimension
agent_speed = [0.5*float(j==0) for j in range(N)] # Leader Speed != 0
visu_frequency = 100 # rviz2 [Hz]
euler_step = 1e-3 # Euler integration step
timer_period = 0.01 # [s]
MAXITERS = 20000 # Max iterations

# HEXAGON distances matrix
L = 2
D = 2*L
H = np.sqrt(3)*L
distances = [[0,     L,      0,    D,     H,    L],
			[L,     0,      L,    0,     D,    0],
			[0,     L,      0,    L,     0,    D],     
			[D,     0,      L,    0,     L,    0],     
			[H,     D,      0,    L,     0,    L],     
			[L,     0,      D,    0,     L,    0]]

planar_formation = True # we want to obtain the formation on a given plane
print('\nDistance Matrix:\n')
pprint(distances)

# Adjacency Matrix
Adj = np.array(distances)>0
Adj = Adj.astype(np.float32).tolist()
print("\nAdjacency Matrix:\n")
pprint(Adj)

Helping_matrix = np.ones((N,N)) - np.eye(N)
print('\nHelping Matrix:\n')
pprint(Helping_matrix)

# Initialize Agents in random positions
x_init = [(np.random.rand(n_x)).tolist() for _ in range(N)] # [x, y, z]
print("\nx_init:\n")
pprint(x_init)

# Initialize Obstacle close to the agents
N_obstacle = 1
# obstacle_pos = [(3*np.random.rand(n_x)).tolist() for _ in range(N_obstacle)] # [x, y, z]
# for list in obstacle_pos:
#      list[-1] = 0.0
# N_obstacle = 0
obstacle_pos = [[3.5,0.0,0.0]] # [x, y, z]
print("\Obstacle positions:\n")
pprint(obstacle_pos)

# define the goal of the navigation
# goal = (4*np.random.rand(n_x)).tolist()
goal = [6.0,0.0,0.0] # [x, y, z]

def generate_launch_description():

    # Define a list in which we append the nodes
    launch_description = [] 

    # Initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('formation_control')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    launch_description.append(
        Node(
            package='rviz2', 
            executable='rviz2', 
            arguments=['-d', rviz_config_file],
            # output='screen',
            # prefix='xterm -title "rviz2" -hold -e'
            ))

    # Launch description for each agent
    for i in range(N):
        N_i = np.nonzero(Adj[:][i])[0].tolist() # Indeces of non zero elements (neighbours)
        N_tot_i = np.nonzero(Helping_matrix[:][i])[0].tolist()

        launch_description.append(
            Node(
                package='formation_control',
                namespace =f'agent_{i}',
                executable='generic_agent',
                parameters=[{ # dictionary for parameters to comunicate to each agent
                                'agent_id': i, 
                                'neigh': N_i,
                                'Adj_row': Adj[i],
                                'all_agents':N_tot_i,
                                'distances_row':distances[i],
                                'planar_formation':planar_formation,
                                'x_init': x_init[i],
                                'max_iters': MAXITERS,
                                'euler_step': euler_step,
                                'timer_period':timer_period,
                                'goal':goal,
                                'agent_speed':agent_speed[i],
                                'N_obstacle':N_obstacle
                                }],
                output='screen',
                prefix=f'xterm -title "agent_{i}" -hold -e',
            ))
        
        launch_description.append(
            Node(
                package='formation_control', 
                namespace='agent_{}'.format(i),
                executable='visualizer', 
                parameters=[{
                                'agent_id': i,
                                'node_frequency': visu_frequency,
                                'agent_or_not':1
                                }],
            ))
    for j in range(N_obstacle):

            launch_description.append(
                Node(
                    package='formation_control',
                    namespace =f'obstacle{j}',
                    executable='generic_obstacle',
                    parameters=[{ # dictionary for parameters to comunicate to each agent
                                    'obstacle_id': j,
                                    'position': obstacle_pos[j],
                                    'max_iters': MAXITERS,
                                    'timer_period':timer_period
                                    }],
                    output='screen',
                    prefix=f'xterm -title "obstacle_{j}" -hold -e',
                ))
            
            launch_description.append(
                Node(
                    package='formation_control', 
                    namespace='obstacle{}'.format(j),
                    executable='visualizer', 
                    parameters=[{
                                    'obstacle_id': j,
                                    'node_frequency': visu_frequency,
                                    'agent_or_not':0
                                    }],
                ))
    return LaunchDescription(launch_description)