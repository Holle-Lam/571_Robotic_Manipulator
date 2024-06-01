import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time
import math

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version p.GUI for graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used to locate the .urdf file

# Load URDF
robot = p.loadURDF("571robotic_arm_final_version/urdf/571robotic_arm_final_version.urdf")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#robot = p.loadURDF("kuka_iiwa/model.urdf")
# You can replace the path above with the actual path of your URDF file

# Set up the camera
width, height, fov = 640, 480, 60
aspect = width / height
near, far = 0.02, 2
view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],
                                                  distance=0.3,
                                                  yaw=0,
                                                  pitch=-40,
                                                  roll=0,
                                                  upAxisIndex=2)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Create a fixed constraint between the robot base and world
p.createConstraint(parentBodyUniqueId=robot,
                   parentLinkIndex=-1,  # -1 indicates the base
                   childBodyUniqueId=-1,  # -1 indicates the world
                   childLinkIndex=-1,
                   jointType=p.JOINT_FIXED,
                   jointAxis=[0, 0, 0],
                   parentFramePosition=[0, 0, 0],
                   childFramePosition=[0, 0, 0])

# Set the camera parameters
camera_distance = 1.  # The distance from the camera to the target position
camera_yaw = 0  # The yaw angle of the camera
camera_pitch = -88  # The pitch angle of the camera
target_position = [0, 0, 0]  # The target position of the camera

# Reset the camera
p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, target_position)

# Get the total number of joints
num_joints = p.getNumJoints(robot)

# The index of the end effector is usually the last link
end_effector_index = num_joints- 1

# Define the maximum number of iterations and the residual threshold
max_iterations = 1000
residual_threshold = 1e-6

# Get the initial state of the end effector
initial_end_effector_state = p.getLinkState(robot, end_effector_index)

# The initial position of the end effector in world coordinates
initial_end_effector_pos = initial_end_effector_state[0]

print(f"The initial position of the end effector is {initial_end_effector_pos}.")

# Define the range of target positions
x_range = np.linspace(-0.3, 0.5, 10)
y_range = np.linspace(-0.3, 0.5, 10)
z_range = np.linspace(0.3, 0.8, 10)

# Iterate over the target positions
for x in x_range:
    for y in y_range:
        for z in z_range:
            # Define the target position
            target_pos = [x, y, z]

            # Print the target position
            print(f"Target position: {target_pos}")

            # Calculate the joint angles
            joint_angles = p.calculateInverseKinematics(robot, end_effector_index, target_pos, maxNumIterations=max_iterations, residualThreshold=residual_threshold)

            # Set the joint angles
            for i in range(min(num_joints, len(joint_angles))):
                p.setJointMotorControl2(bodyUniqueId=robot,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_angles[i])

            # Step the simulation forward
            for _ in range(800):
                p.stepSimulation()

            # Get the new state of the end effector
            end_effector_state = p.getLinkState(robot, end_effector_index)

            # The new position of the end effector in world coordinates
            new_end_effector_pos = end_effector_state[0]

            # Print the current end effector position
            print(f"Current end effector position: {new_end_effector_pos}")

            # Compare the new end effector position with the target position
            if np.allclose(new_end_effector_pos, target_pos, atol=0.1):
                print(f"For target position {target_pos}, the end effector reached the target position.")