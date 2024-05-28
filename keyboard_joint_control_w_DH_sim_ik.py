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

from scipy.optimize import minimize
import sympy as sp
from sympy import symbols, cos, sin, pi, Matrix


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

mod_real_robot = False
mod_dh_sim = True
mod_test = False # allow some test code in main control loop

# The joint can be defined as rotational with 'r', or translational with 't', if there is an ebd-effector frame, you can also add one more row as dh parameter and joint type to be 'f' (fixed joint)
# By defination, the d or theta of the DH parameters will be variable and the other one will be constant
joint_type = ['r', 'r', 'r', 'r', 'f']
joint_pos_init = [0, 0, 0, 0]
joint_pos_home = [0, 0, 0, 0]  # this is the home position, when pressing home key, the robot arm will be reset to home position

# DH Parameters given in order of [a, alpha, d, theta] for each joint, angles should be given in degree
# If the joint is rotational, then the 4th entry will be variable and the number given here will be treat as offset of that joint. This is the same if the joint is translational so that d is variable

# New DH with all zero a value, which makes close-form IK easier
dh_parameters = [[0.0, 0.0, 98, 0.0],
 [90.0, 0.0, 107, 0.0],
 [45.0, 0.0, -124, 180.0],
 [90.0, 0.0, 0.0, 0.0],
 [0.0, 0.0, 168, 90]]


# 
joint_speed = [3, 3, 3, 3, 3, 3]
ee_speed = [5, 5, 5]

if mod_real_robot: 
  keyboard_increment = 3

deg2rad = np.pi/180.0
rad2deg = 180.0/np.pi

plt_axis_limit = 200  # This value should be changed based on the maximum lenth of the robot arm for better visualization
plt_x_lim = [-plt_axis_limit, plt_axis_limit]
plt_y_lim = [-plt_axis_limit, plt_axis_limit]
plt_z_lim = [0, plt_axis_limit]

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
    print('End-effector Control')
    print('[End-effector X +]: l | [End-effector X -]: .')
    print('[End-effector Y +]: , | [End-effector Y -]: /')
    print('[End-effector Z +]: k | [End-effector Z -]: m')
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
        ax.quiver(current_position[0, 3], current_position[1, 3], current_position[2, 3], current_position[0, 0], current_position[1, 0], current_position[2, 0], color='y', length=0.1*plt_axis_limit, label = 'Joint '+str(i+1)+' ' + jpos)  # X-axis in blue, , length=0.1*plt_axis_limit, width=0.005*plt_axis_limit
        ax.quiver(current_position[0, 3], current_position[1, 3], current_position[2, 3], current_position[0, 2], current_position[1, 2], current_position[2, 2], color='r', length=0.1*plt_axis_limit)  # Z-axis in red

        previous_position = current_position
    plt.title('End-effector Location (x, y, z): ' + str([round(current_position[0, 3],2), round(current_position[1, 3],2), round(current_position[2, 3],2)]))
    ax.legend()
    return current_position
  

def normalize_angle(angle):
    # Normalize the angle to the range [0, 360)
    angle = angle % 360
    # Convert it to the range (-180, 180]
    if angle > 180:
        angle -= 360
    return angle
  
# numerical forward kinematics that is used for solving inverse kinematics
def forward_kinematics_num(joint_positions, dh_params):
    """
    Compute the forward kinematics using the DH parameters and joint angles.
    """
    current_position = base_frame * 1
    for i, (alpha, a, d, theta) in enumerate(dh_parameters):
        alpha = alpha*deg2rad   # need to convert alpha to rad
        if joint_type[i] == 'r':
          theta = theta*deg2rad + joint_positions[i]    
        
        elif joint_type[i] == 't':
          d = d + joint_positions[i]

        elif joint_type[i] == 'f':
          theta = theta*deg2rad   # do nothing, but still need to convert theta from degree to rad
        else:
          print('[ERR]: Unknown joint type')
        
        # transformation_matrix = dh_to_transformation_matrix(alpha, a, d, theta)
        # Update current position based on transformation matrix
        current_position = current_position @ np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)],
            [0, 0, 0, 1]])
    # print(current_position)
    # print(joints)
    # print('-------------------------------')
    return current_position
  
# symbolic forward kinematics, will be used to solve the numerical inverse kinematics 
def forward_kinematics_sym(joint_positions, dh_params):
    """
    Compute the forward kinematics using the DH parameters and joint angles.
    """
    # T = Matrix.eye(4)
    T = Matrix(base_frame)
    for i, (alpha, a, d, theta) in enumerate(dh_parameters):
        alpha = alpha*deg2rad   # need to convert alpha to rad
        if joint_type[i] == 'r':       
          # print(len(joint_positions))
          # print(joint_positions)
          theta = theta*deg2rad + joint_positions[i]    
        
        elif joint_type[i] == 't':
          d = d + joint_positions[i]

        elif joint_type[i] == 'f':
          theta = theta*deg2rad   # do nothing, but still need to convert theta from degree to rad

        else:
          print('[ERR]: Unknown joint type')
        # DH Transformation matrix
        T = T * Matrix([
            [cos(theta), -sin(theta), 0, a],
            [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
            [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
            [0, 0, 0, 1]
            ])
    # print(T)
    # print(joint_positions)
    # print('-------------------------------')
    return T
  
def objective_function(joints, desired_pos, dh_params):
    """
    The objective function that calculates the error between the current end-effector
    position and the desired position.
    """
    # print(joints)
    fk = forward_kinematics_num(joints, dh_params)
    # Position from forward kinematics
    current_pos = fk[:3, 3]
    # print('iter fk: ', fk)
    # print('iter joints: ', joints)
    # print('iter pos: ', current_pos)
    # Calculate error
    error = np.linalg.norm(np.array(current_pos).astype(np.float64).flatten() - np.array(desired_pos))
    # print('current err: ', error)
    return error

def inverse_kinematics(dh_params, desired_pos, initial_guess):
    # inspired by ChatGPT: https://chat.openai.com/share/88b7d977-e666-4705-b4cd-42edc23a91c3
      
    initial_guess = np.array(initial_guess) * deg2rad
    res = minimize(objective_function, initial_guess, args=(desired_pos, dh_params), method='SLSQP') # in terms of SLSQP works well, however BFGS does not work
    if res.success:
        # return res.x * rad2deg
        return [normalize_angle(angle) for angle in res.x * rad2deg]
    else:
        print("Inverse kinematics did not converge")
        return initial_guess * rad2deg
      
# close form inverse kinematics solution, eps_0 is a small number that is used to determine if certain values are too close to 0 and cause singularity
# current jpos should be 
def inverse_kinematics_close_form(dh_params, desired_pos, current_jpos, eps_0 = 1e-2):
  x, y, z = desired_pos
  l1 = 97.8
  l2 = 106.928
  l3 = -124.481
  l4 = 167.800
  sqrt2 = np.sqrt(2) # this will be commonly used
  c3 = (x**2 + y**2 + (z-l1)**2 - l2**2 - l3**2 - l4**2 - sqrt2*l2*l3) / (-sqrt2*l2*l4)
  print('c3: ', c3)
  if c3 > 1 or c3 < -1:  # if cos(th3) is not within [-1, 1], then inverse kinematics has no solution, this could mainly be caused by out of workspace
    print('bad c3')
    return None
  # 2 solutions of th3
  th3_1 = np.arccos(c3)
  th3_2 = -np.arccos(c3)
  
  k3 = 0.5*sqrt2*l3 + 0.5*sqrt2*l4*c3
  k4_1 = l4 * np.sin(th3_1)
  k4_2 = l4 * np.sin(th3_2)
  
  h1_sqr = x**2 + y**2 - (l2 + 0.5*sqrt2*l3 - 0.5*sqrt2*l4*c3)**2
  if h1_sqr < 0:
    print('bad h1')
    return None  # if square is smaller than 0, IK has no solution
  h1_1 = np.sqrt(h1_sqr)
  h1_2 = -np.sqrt(h1_sqr)
  h2 = z - l1
  
  # th2 has 4 solutions based on combination of k4 and h1
  k4 = k4_1 *1.0
  h1 = h1_1 *1.0
  s2 = (-k3*h1 - k4*h2) / (-k3**2 - k4**2)
  c2 = (-k4*h1 + k3*h2) / (-k3**2 - k4**2)
  if s2>1 or s2<-1 or c2>1 or c2<-1:   # if sine or cosine values are not reasonable, IK has no solution
    print('bad th2')
    # return None
  print('s2:', s2)
  print('c2:', c2)
  th2_1 = np.arctan2(s2, c2)
  #---
  k4 = k4_1 *1.0
  h1 = h1_2 *1.0
  s2 = (-k3*h1 - k4*h2) / (-k3**2 - k4**2)
  c2 = (-k4*h1 + k3*h2) / (-k3**2 - k4**2)
  if s2>1 or s2<-1 or c2>1 or c2<-1:   # if sine or cosine values are not reasonable, IK has no solution
    print('bad th2')
    # return None
  th2_2 = np.arctan2(s2, c2)
  #---
  k4 = k4_2 *1.0
  h1 = h1_1 *1.0
  s2 = (-k3*h1 - k4*h2) / (-k3**2 - k4**2)
  c2 = (-k4*h1 + k3*h2) / (-k3**2 - k4**2)
  if s2>1 or s2<-1 or c2>1 or c2<-1:   # if sine or cosine values are not reasonable, IK has no solution
    print('bad th2')
    # return None
  th2_3 = np.arctan2(s2, c2)
  #---
  k4 = k4_2 *1.0
  h1 = h1_2 *1.0
  s2 = (-k3*h1 - k4*h2) / (-k3**2 - k4**2)
  c2 = (-k4*h1 + k3*h2) / (-k3**2 - k4**2)
  if s2>1 or s2<-1 or c2>1 or c2<-1:   # if sine or cosine values are not reasonable, IK has no solution
    print('bad th2')
    # return None
  th2_4 = np.arctan2(s2, c2)
  #---
  
  # th1 has 4 solutions based on th2 and th3, details can be found in the multiple solution tree
  th3 = th3_1 *1.0
  th2 = th2_1 *1.0
  k1 = l2 + 0.5*sqrt2*l3-0.5*sqrt2*l4*np.cos(th3)
  k2 = 0.5*sqrt2*l3*np.sin(th2) + l4*np.cos(th2)*np.sin(th3) + 0.5*sqrt2*l4*np.sin(th2)*np.cos(th3)
  s1 = (-k1*x - k2*y) / (-k1**2 - k2**2)
  c1 = (-k2*x + k1*y) / (-k1**2 - k2**2)
  if s1>1 or s1<-1 or c1>1 or c1<-1:   # if sine or cosine values are not reasonable, IK has no solution
    print('bad th1')
    # return None
  th1_1 = np.arctan2(s1, c1)
  #---
  th3 = th3_1 *1.0
  th2 = th2_2 *1.0
  k1 = l2 + 0.5*sqrt2*l3-0.5*sqrt2*l4*np.cos(th3)
  k2 = 0.5*sqrt2*l3*np.sin(th2) + l4*np.cos(th2)*np.sin(th3) + 0.5*sqrt2*l4*np.sin(th2)*np.cos(th3)
  s1 = (-k1*x - k2*y) / (-k1**2 - k2**2)
  c1 = (-k2*x + k1*y) / (-k1**2 - k2**2)
  if s1>1 or s1<-1 or c1>1 or c1<-1:   # if sine or cosine values are not reasonable, IK has no solution
    print('bad th1')
    # return None
  th1_2 = np.arctan2(s1, c1)
  #---
  th3 = th3_2 *1.0
  th2 = th2_3 *1.0
  k1 = l2 + 0.5*sqrt2*l3-0.5*sqrt2*l4*np.cos(th3)
  k2 = 0.5*sqrt2*l3*np.sin(th2) + l4*np.cos(th2)*np.sin(th3) + 0.5*sqrt2*l4*np.sin(th2)*np.cos(th3)
  s1 = (-k1*x - k2*y) / (-k1**2 - k2**2)
  c1 = (-k2*x + k1*y) / (-k1**2 - k2**2)
  if s1>1 or s1<-1 or c1>1 or c1<-1:   # if sine or cosine values are not reasonable, IK has no solution
    print('bad th1')
    # return None
  th1_3 = np.arctan2(s1, c1)
  #---
  th3 = th3_2 *1.0
  th2 = th2_4 *1.0
  k1 = l2 + 0.5*sqrt2*l3-0.5*sqrt2*l4*np.cos(th3)
  k2 = 0.5*sqrt2*l3*np.sin(th2) + l4*np.cos(th2)*np.sin(th3) + 0.5*sqrt2*l4*np.sin(th2)*np.cos(th3)
  s1 = (-k1*x - k2*y) / (-k1**2 - k2**2)
  c1 = (-k2*x + k1*y) / (-k1**2 - k2**2)
  if s1>1 or s1<-1 or c1>1 or c1<-1:   # if sine or cosine values are not reasonable, IK has no solution
    print('bad th1')
    # return None
  th1_4 = np.arctan2(s1, c1)
  #---
  
  offset_j1 = dh_params[0][3] *deg2rad
  offset_j2 = dh_params[1][3] *deg2rad
  offset_j3 = dh_params[2][3] *deg2rad
  
  # print('th3_1:' , th3_1 *rad2deg )
  # print('th3_2:' , th3_2 *rad2deg )
  
  ik_solution_1 = [normalize_angle((th1_1-offset_j1)*rad2deg), normalize_angle((th2_1-offset_j2)*rad2deg), normalize_angle((th3_1-offset_j3)*rad2deg)]
  ik_solution_2 = [normalize_angle((th1_2-offset_j1)*rad2deg), normalize_angle((th2_2-offset_j2)*rad2deg), normalize_angle((th3_1-offset_j3)*rad2deg)]
  ik_solution_3 = [normalize_angle((th1_3-offset_j1)*rad2deg), normalize_angle((th2_3-offset_j2)*rad2deg), normalize_angle((th3_2-offset_j3)*rad2deg)]
  ik_solution_4 = [normalize_angle((th1_4-offset_j1)*rad2deg), normalize_angle((th2_4-offset_j2)*rad2deg), normalize_angle((th3_2-offset_j3)*rad2deg)]
  
  ik_solutions = np.array([ik_solution_1, ik_solution_2, ik_solution_3, ik_solution_4])
  
  return ik_solutions
  
  
      
# return the Jacobian matrix for velocity control
def jacobian_matrix_num(joint_positions, dh_params):
    # inspired by ChatGPT: https://chat.openai.com/share/5487a664-138d-4644-a0d4-14f25647525a
    
    T = Matrix(base_frame)
    # Symbol for joint angle   
    
    joint_symbols = []
    # Subsequent links
    for i, (alpha, a, d, theta) in enumerate(dh_parameters):
        alpha = alpha*deg2rad   # need to convert alpha to rad
        if joint_type[i] == 'r':       
          theta = sp.symbols('J_' + str(i))  # if joint is rotational, replace theta by symbol
          joint_symbols.append(theta)
      
        elif joint_type[i] == 't':
          d = sp.symbols('J_' + str(i))  # if joint is rotational, replace theta by symbol
          joint_symbols.append(d)

        elif joint_type[i] == 'f':
          theta = theta*deg2rad   # do nothing, but still need to convert theta from degree to rad

        else:
          print('[ERR]: Unknown joint type')
        # DH Transformation matrix
        T = T * Matrix([
            [cos(theta), -sin(theta), 0, a],
            [sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
            [sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), cos(alpha)*d],
            [0, 0, 0, 1]
            ])
    
    # Position of the end-effector
    position = T[:3, 3]
    
    # Jacobian computation
    J = sp.zeros(3, len(joint_positions))
    for i in range(len(joint_positions)):
        J[:, i] = sp.diff(position, joint_symbols[i])
    
    # Substitute joint positions into Jacobian
    joint_dict = {}
    for i, (alpha, a, d, theta) in enumerate(dh_parameters):
      if joint_type[i] == 'r':       
        theta = theta*deg2rad + joint_positions[i]*deg2rad    
        joint_dict[joint_symbols[i]] = theta
      elif joint_type[i] == 't':
        d = d + joint_positions[i]
        joint_dict[joint_symbols[i]] = d
      elif joint_type[i] == 'f':
        _ = 1 # do nothing
        
    J = J.subs(joint_dict)
    # print(J)
    return np.array(J).astype(float)

# take desired end-effector velocity as input and output joint velocity. The vel_d should be given as np.array with shape of (3,)
def jacobian_velocity(vel_d, joint_positions, dh_params):
  Jac = jacobian_matrix_num(joint_positions, dh_params)
  j_vel = np.linalg.pinv(Jac).dot(vel_d.T)
  return j_vel * rad2deg




# main control loop
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
  
  
previous_position = np.array(base_frame)
# init the end-effector position based on the initial joint positions
for i, (alpha, a, d, theta) in enumerate(dh_parameters):

    if joint_type[i] == 'r':
      theta = theta + joint_pos_init[i]    
      jpos = str(round(joint_pos_init[i], 2))
    elif joint_type[i] == 't':
      d = d + joint_pos_init[i]
      jpos = str(round(joint_pos_init[i], 2))
    elif joint_type[i] == 'f':
      jpos = 'fixed'     # fixed last joint for the end-effector frame
    else:
      print('[ERR]: Unknown joint type')
    
    transformation_matrix = dh_to_transformation_matrix(alpha, a, d, theta)

    # Update current position based on transformation matrix
    current_position = previous_position @ transformation_matrix
ee_pos_cur = current_position[:3,3]*1.0 # initialize the current end-effector position
ee_pos_d = current_position[:3,3]*1.0 # initialize the desired end-effector position 

working = 1
command = False
ik_command = False   
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
        
    elif input_key == '5':
        print_no_newline(" Moving: Joint 4 +++         ")
        if mod_real_robot:
          goals[4] += keyboard_increment
        if mod_dh_sim:
          joint_positions[4] += joint_speed[4]
        command = True
        
    elif input_key == 't':
        print_no_newline(" Moving: Joint 4 ---         ")
        if mod_real_robot:
          goals[4] -= keyboard_increment
        if mod_dh_sim:
          joint_positions[4] -= joint_speed[4]
        command = True
        
    elif input_key == '6':
        print_no_newline(" Moving: Joint 4 +++         ")
        if mod_real_robot:
          goals[5] += keyboard_increment
        if mod_dh_sim:
          joint_positions[5] += joint_speed[5]
        command = True
        
    elif input_key == 'y':
        print_no_newline(" Moving: Joint 4 ---         ")
        if mod_real_robot:
          goals[5] -= keyboard_increment
        if mod_dh_sim:
          joint_positions[5] -= joint_speed[5]
        command = True
    
    elif input_key == 'h':
        print_no_newline(" Homing....                  ")
        if mod_real_robot:
          goals = np.array(joint_pos_home)
        if mod_dh_sim:
          joint_positions = joint_pos_home * 1
        command = True
        
    #Keys for cartesian based control-----------------------------------------
    elif input_key == 'l':
        print_no_newline(" Moving: x +++              ")
        ee_pos_d = ee_pos_cur + [ee_speed[0], 0, 0]
        # print('ee_pos_cur', ee_pos_cur)
        # print('ee_pos_d', ee_pos_d)
        command = True
        ik_command = True
    elif input_key == '.':
        print_no_newline(" Moving: x ---              ")
        ee_pos_d = ee_pos_cur - [ee_speed[0], 0, 0]
        command = True
        ik_command = True
    elif input_key == ',':
        print_no_newline(" Moving: y +++              ")
        ee_pos_d = ee_pos_cur + [0, ee_speed[1], 0]
        command = True
        ik_command = True
    elif input_key == '/':
        print_no_newline(" Moving: y ---              ")
        ee_pos_d = ee_pos_cur - [0, ee_speed[1], 0]
        command = True
        ik_command = True
    elif input_key == 'k':
        print_no_newline(" Moving: z +++              ")
        ee_pos_d = ee_pos_cur + [0, 0, ee_speed[2]]
        command = True
        ik_command = True
    elif input_key == 'm':
        print_no_newline(" Moving: z ---              ")
        ee_pos_d = ee_pos_cur - [0, 0, ee_speed[2]]
        command = True
        ik_command = True
        
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

    
    if command:
        if mod_real_robot:
        # make sure the goals is within joint limit
          if ik_command:
            goals = inverse_kinematics(dh_parameters, ee_pos_d, joint_positions)
            # print(goals)
          goals = np.clip(goals, RC.servo_angle_min, RC.servo_angle_max) 
          sys.stdout.write("\033[1B") # move curser down
          RC.joints_goto(goals, speeds)
          sys.stdout.write("\033[1A") # move curser up
          
        
        if mod_dh_sim:
          sys.stdout.write("\033[1B") # move curser down
          sys.stdout.write("\033[1A") # move curser up
          if ik_command:
            joint_positions = inverse_kinematics(dh_parameters, ee_pos_d, joint_positions)
            joint_positions_cf = inverse_kinematics_close_form(dh_parameters, ee_pos_d, joint_positions, eps_0 = 1e-2)
            print("--------------------------------")
            print('numerical IK')
            print(joint_positions)          
            print('close form IK')
            print(joint_positions_cf)
          # ani = FuncAnimation(fig, update_frame, frames=1, fargs=(dh_parameters, base_frame, joint_positions), interval=100)
          plot_frame(frame, dh_parameters, base_frame, joint_positions)
          plt.show(block = False)
          plt.pause(0.01)
          frame += 1
        if mod_test:
          joint_vel = jacobian_velocity(np.array([0,0,-1]), joint_positions, dh_parameters)
          # print(joint_vel)
          
        ee_pos_cur = plot_frame(frame, dh_parameters, base_frame, joint_positions)[:3,3]
        
        # print(jacobian_matrix_num(joint_positions, dh_parameters))
        command = False
        if ik_command:
          ik_command = False
      

