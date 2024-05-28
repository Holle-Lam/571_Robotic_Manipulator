# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:35:15 2024

@author: 75678
"""

import time
import numpy as np

from scipy.optimize import minimize
import sympy as sp
from sympy import symbols, cos, sin, pi, Matrix

# The joint can be defined as rotational with 'r', or translational with 't', if there is an ebd-effector frame, you can also add one more row as dh parameter and joint type to be 'f' (fixed joint)
# By defination, the d or theta of the DH parameters will be variable and the other one will be constant
joint_type = ['r', 'r', 'r', 'r']

# DH Parameters given in order of [a, alpha, d, theta] for each joint, angles should be given in degree
# If the joint is rotational, then the 4th entry will be variable and the number given here will be treat as offset of that joint. This is the same if the joint is translational so that d is variable
dh_parameters = [[0.0, 0.0, 98, 0.0],
 [90.0, 0.0, 107, 0.0],
 [45.0, 0.0, -124, 180.0],
 [90.0, 0.0, 168.0, 90.0]]


# Base Frame (Identity Matrix for simplicity), by defult (base_frame = np.eye(4)) it is coincide with the plot axis and origin
# If you want to change the base frame, you can change the transfromation matrix below
base_frame = np.eye(4)

deg2rad = np.pi/180.0
rad2deg = 180.0/np.pi

def normalize_angle(angle):
    # Normalize the angle to the range [0, 360)
    angle = angle % 360
    # Convert it to the range (-180, 180]
    if angle > 180:
        angle -= 360
    return angle

# numerical forward kinematics that is used for solving inverse kinematics
def forward_kinematics_num(joint_positions, dh_params, base_frame):
    """
    Compute the forward kinematics using the DH parameters and joint angles.
    """
    current_position = base_frame * 1
    for i, (alpha, a, d, theta) in enumerate(dh_parameters):
        alpha = alpha*deg2rad   # need to convert alpha to rad
        if joint_type[i] == 'r':
          theta = theta*deg2rad + joint_positions[i]*deg2rad    
        
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
    return current_position
  
def objective_function(joints, desired_pos, dh_params, base_frame):
    """
    The objective function that calculates the error between the current end-effector
    position and the desired position.
    """
    print(joints)
    fk = forward_kinematics_num(joints, dh_params, base_frame)
    # Position from forward kinematics
    current_pos = fk * 1

    error_loc = np.linalg.norm(np.array(current_pos[:3, 3]).astype(np.float64).flatten() - np.array(desired_pos[:3, 3].flatten()))
    error_ori = 10 * np.sum(np.abs(np.array(current_pos[:3, :3]).astype(np.float64).flatten() - np.array(desired_pos[:3, :3].flatten()))) # amplify the orientation error so that the solve do not sacrifice accuracy in orientation
    print('location err: ', error_loc)
    print('orientation err: ', error_ori)
    error = error_loc + error_ori

    return error

def inverse_kinematics(dh_params, desired_pos, initial_guess, base_frame):
    # inspired by ChatGPT: https://chat.openai.com/share/88b7d977-e666-4705-b4cd-42edc23a91c3    
    initial_guess = np.array(initial_guess) 
    res = minimize(objective_function, initial_guess, args=(desired_pos, dh_params, base_frame), method='SLSQP') # in terms of SLSQP works well, however BFGS does not work
    if res.success:
        # return res.x * rad2deg
        return np.array([normalize_angle(angle) for angle in res.x])
    else:
        print("Inverse kinematics did not converge")
        return initial_guess * rad2deg

# the joint positions should be given in degree
# test_joint_position = [50, 60, 70, 80]
test_joint_position = np.array([np.random.uniform(-180, 180) for i in range(4)])
fk_rst = forward_kinematics_num(joint_positions = test_joint_position, dh_params = dh_parameters, base_frame = base_frame)

ik_joint_solution = inverse_kinematics(dh_params = dh_parameters, 
                                       desired_pos = fk_rst, 
                                       initial_guess = test_joint_position + np.array([np.random.uniform(-50, 50) for i in range(4)]),
                                       base_frame = base_frame)

np.set_printoptions(precision=3, suppress=True)
print('--------------------------')
print('Ground truth joint position:')
print(test_joint_position)
print('Ground truth end-effector pose:')
print(fk_rst)

print('--------------------')
print('Inverse kinematic solved joint position:')
print(ik_joint_solution)
print('End-effector pose based on IK solution')
print(forward_kinematics_num(joint_positions = ik_joint_solution, dh_params = dh_parameters, base_frame = base_frame))


