"""""""""""""""""""""""""""""

University of Washington, 2024

Author: Tin Chiang

Note: Modified code from Haonan Peng's Raven keyboard controller

Original code: https://github.dev/uw-biorobotics/raven2_CRTK_Python_controller/blob/main/python_controller/run_r2_keyboard_controller.py
"""""""""""""""""""""""""""""

import time
import numpy as np
import keyboard
import sys, os


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

mod_real_robot = True
mod_dh_sim = True

# The joint can be defined as rotational with 'r', or translational with 't', if there is an ebd-effector frame, you can also add one more row as dh parameter and joint type to be 'f' (fixed joint)
# By defination, the d or theta of the DH parameters will be variable and the other one will be constant
joint_type = ['r', 'r', 'r', 'r', 'f']
joint_pos_init = [0, 0, 0, 0]

# DH Parameters given in order of [a, alpha, d, theta] for each joint, angles should be given in degree
# If the joint is rotational, then the 4th entry will be variable and the number given here will be treat as offset of that joint. This is the same if the joint is translational so that d is variable
dh_parameters = [
    [0, 0, 62.8, 90],    # Joint 1
    [90, 0, -5.1, 90],   # Joint 2
    [-90, 105, 0, 0],   # Joint 3
    [90, 27, 4.9, 90],   # Joint 4
    [0, 0, 10, 0]        # end-effector joint, fixed
]

dh_parameters = [[0.000, 0.000, 84.504, 0.196],
 [90.000, 20.174, 106.928, -1.793],
 [45.000, 0.000, -124.481, 179.864],
 [90.000, 0.000, 0.000, 0.000],
 [0.000, 0.000, 167.800, 90.002]]

# 
joint_speed = [1, 1, 1, 1]
keyboard_increment = 1

plt_axis_limit = 100  # This value should be changed based on the maximum lenth of the robot arm for better visualization
plt_x_lim = [-plt_axis_limit, plt_axis_limit]
plt_y_lim = [-plt_axis_limit, plt_axis_limit]
plt_z_lim = [0, 2*plt_axis_limit]

# Base Frame (Identity Matrix for simplicity), by defult (base_frame = np.eye(4)) it is coincide with the plot axis and origin
# If you want to change the base frame, you can change the transfromation matrix below
base_frame = np.eye(4)



if mod_real_robot:
  from robot_controller import robot_controller


def print_manu():
    print('  ')
    print('-----------------------------------------')
    print('EE543 Arm DH Simulation Keyboard Controller:')
    
    print('Joint Control')
    print('[Exit]: 9')
    print('[Joint 1 +]: 1 | [Joint 1 -]: q')
    print('[Joint 2 +]: 2 | [Joint 2 -]: w')
    print('[Joint 3 +]: 3 | [Joint 3 -]: e')
    print('[Joint 4 +]: 4 | [Joint 4 -]: r')
    print('-----------------------------------------')
    print('View Control')
    print('[View Elev +]: s  | [View Elev -]: x')
    print('[View Azim +]: z  | [View Azim -]: c')

    print('-----------------------------------------')
    print('-----------------------------------------')
    print('Current command:\n')
    return None

def print_no_newline(string):
    sys.stdout.write("\r" + string)
    sys.stdout.flush()
    return None
  
def dh_to_transformation_matrix(alpha, a, d, theta):
    alpha_rad = np.radians(alpha)
    theta_rad = np.radians(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0, a],
        [np.sin(theta_rad)*np.cos(alpha_rad), np.cos(theta_rad)*np.cos(alpha_rad), -np.sin(alpha_rad), -d*np.sin(alpha_rad)],
        [np.sin(theta_rad)*np.sin(alpha_rad), np.cos(theta_rad)*np.sin(alpha_rad), np.cos(alpha_rad), d*np.cos(alpha_rad)],
        [0, 0, 0, 1]
    ])
  
def plot_frame(frame, dh_parameters, base_frame, joint_positions):
    ax.clear()
    ax.view_init(elev=plt_elev, azim=plt_azim)
    ax.set_xlim(plt_x_lim)
    ax.set_ylim(plt_y_lim)
    ax.set_zlim(plt_z_lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Starting point (base frame)
    current_position = np.array(base_frame)
    previous_position = np.array(base_frame)
    
    # Draw base frame axes
    ax.quiver(current_position[0, 3], current_position[1, 3], current_position[2, 3], current_position[0, 0], current_position[1, 0], current_position[2, 0], color='y', length=0.1*plt_axis_limit)  # X-axis in blue, , length=0.1*plt_axis_limit, width=0.005*plt_axis_limit
    ax.quiver(current_position[0, 3], current_position[1, 3], current_position[2, 3], current_position[0, 2], current_position[1, 2], current_position[2, 2], color='r', length=0.1*plt_axis_limit)  # Z-axis in red

    for i, (alpha, a, d, theta) in enumerate(dh_parameters):

        if joint_type[i] == 'r':
          theta = theta + joint_positions[i]    
          jpos = str(round(joint_positions[i], 2))
        elif joint_type[i] == 't':
          d = d + joint_positions[i]
          jpos = str(round(joint_positions[i], 2))
        elif joint_type[i] == 'f':
          jpos = 'fixed'     # fixed last joint for the end-effector frame
        else:
          print('[ERR]: Unknown joint type')
        
        transformation_matrix = dh_to_transformation_matrix(alpha, a, d, theta)

        # Update current position based on transformation matrix
        current_position = previous_position @ transformation_matrix

        # Draw links
        ax.plot([previous_position[0, 3], current_position[0, 3]], [previous_position[1, 3], current_position[1, 3]], [previous_position[2, 3], current_position[2, 3]], 'k-', linewidth = 5)
        
        # Draw coordinate frame axes for each link
        ax.quiver(current_position[0, 3], current_position[1, 3], current_position[2, 3], current_position[0, 0], current_position[1, 0], current_position[2, 0], color='y', length=0.1*plt_axis_limit, label = 'Joint '+str(i+1)+': ' + jpos)  # X-axis in blue, , length=0.1*plt_axis_limit, width=0.005*plt_axis_limit
        ax.quiver(current_position[0, 3], current_position[1, 3], current_position[2, 3], current_position[0, 2], current_position[1, 2], current_position[2, 2], color='r', length=0.1*plt_axis_limit)  # Z-axis in red

        previous_position = current_position
    plt.title('End-effector Location (x, y, z): ' + str([round(current_position[0, 3],2), round(current_position[1, 3],2), round(current_position[2, 3],2)]), fontsize = 20)
    ax.legend(fontsize = 20, loc=1)
    return None

if mod_real_robot:
  # init the Robot Controller
  RC = robot_controller()
  RC.communication_begin()
  
  # Force homing the robot
  RC.joints_homing()
  
  
  
  goals = np.zeros(RC.joint_num)
  speeds = np.ones(RC.joint_num) * 80 # deg/s

if mod_dh_sim:
# Setup the plot
  fig = plt.figure(figsize=(15,15))
  fig.clf()
  ax = fig.add_subplot(111, projection='3d')
  
  working = 1
  command = False
 
  plt_elev = 45
  plt_azim = 45
  
  joint_positions = joint_pos_init
  
  # ani = FuncAnimation(fig, update_frame, frames=1, fargs=(dh_parameters, base_frame, joint_positions), interval=100)
  frame = 0
  plt.show()

working = 1
command = False
print_manu()

while working==1:

    #get the keyboard input
    input_key = keyboard.read_event().name

    if input_key == '9':
        # RC.communication_end()
        os.system('cls' if os.name == 'nt' else 'clear')
        sys.exit('Closing Keyboard controller')
        

    elif input_key == '1':
        print_no_newline(" Moving: Joint 1 +++         ")
        if mod_real_robot:
          goals[0] += keyboard_increment
        if mod_dh_sim:
          joint_positions[0] += joint_speed[0]
        command = True


    elif input_key == 'q':
        print_no_newline(" Moving: Joint 1 ---         ")
        if mod_real_robot:
          goals[0] -= keyboard_increment
        if mod_dh_sim:
          joint_positions[0] -= joint_speed[0]          
        command = True

    elif input_key == '2':
        print_no_newline(" Moving: Joint 2 +++         ")
        if mod_real_robot:
          goals[1] += keyboard_increment
        if mod_dh_sim:
          joint_positions[1] += joint_speed[1]
        command = True
              
    elif input_key == 'w':
        print_no_newline(" Moving: Joint 2 ---         ")
        if mod_real_robot:
          goals[1] -= keyboard_increment
        if mod_dh_sim:
          joint_positions[1] -= joint_speed[1]
        command = True

    elif input_key == '3':
        print_no_newline(" Moving: Joint 3 +++         ")
        if mod_real_robot:
          goals[2] += keyboard_increment
        if mod_dh_sim:
          joint_positions[2] += joint_speed[2]
        command = True
        
    elif input_key == 'e':
        print_no_newline(" Moving: Joint 3 ---         ")
        if mod_real_robot:
          goals[2] -= keyboard_increment
        if mod_dh_sim:
          joint_positions[2] -= joint_speed[2]
        command = True

    elif input_key == '4':
        print_no_newline(" Moving: Joint 4 +++         ")
        if mod_real_robot:
          goals[3] += keyboard_increment
        if mod_dh_sim:
          joint_positions[3] += joint_speed[3]
        command = True
        
    elif input_key == 'r':
        print_no_newline(" Moving: Joint 4 ---         ")
        if mod_real_robot:
          goals[3] -= keyboard_increment
        if mod_dh_sim:
          joint_positions[3] -= joint_speed[3]
        command = True
    
    elif input_key == 'h':
        print_no_newline(" Homing....                  ")
        if mod_real_robot:
          goals = RC.robot_homing_joint_poses.copy()
        if mod_dh_sim:
          joint_positions = [0, 0, 0, 0]
        command = True
        
    # keys to adjust view angle    
    elif input_key == 's':
        print_no_newline(" View Elev +++                  ")
        plt_elev += 5
        command = True
      
    elif input_key == 'x':
        print_no_newline(" View Elev ---                  ")
        plt_elev -= 5
        command = True
        
    elif input_key == 'z':
        print_no_newline(" View Azim +++                  ")
        plt_azim += 5
        command = True
        
    elif input_key == 'c':
        print_no_newline(" View Azim ---                  ")
        plt_azim -= 5
        command = True
        

    else:
        print_no_newline(' Unknown command             ')

    sys.stdout.write("\033[1B") # move curser down
    sys.stdout.write("\033[1A") # move curser up
    if command:
        if mod_real_robot:
        # make sure the goals is within joint limit
          goals = np.clip(goals, RC.servo_angle_min, RC.servo_angle_max) 
          
          RC.joints_goto(goals, speeds)
          
          command = False
        
        if mod_dh_sim:
          command = False
          # ani = FuncAnimation(fig, update_frame, frames=1, fargs=(dh_parameters, base_frame, joint_positions), interval=100)
          plot_frame(frame, dh_parameters, base_frame, joint_positions)
          plt.show(block = False)
          plt.pause(0.01)
          frame += 1

