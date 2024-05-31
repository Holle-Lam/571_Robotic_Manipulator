import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time
import math

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version p.GUI for graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used to locate the .urdf file

# Load URDF
robot = p.loadURDF("571_robotic_arm_urdf_mesh/urdf/571_robotic_arm_urdf_mesh.urdf")

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
camera_distance = 0.5  # The distance from the camera to the target position
camera_yaw = 0  # The yaw angle of the camera
camera_pitch = -40  # The pitch angle of the camera
target_position = [0, 0, 0]  # The target position of the camera

# Reset the camera
p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, target_position)

# Get the total number of joints
num_joints = p.getNumJoints(robot)

# The index of the end effector is usually the last link
end_effector_index = num_joints - 1

# Get the state of the end effector
end_effector_state = p.getLinkState(robot, end_effector_index)

# The position and orientation of the end effector in world coordinates
end_effector_pos = end_effector_state[0]
end_effector_orn = end_effector_state[1]

print("End effector position:", end_effector_pos)
print("End effector orientation:", end_effector_orn)

# Define the target position and orientation for the end effector
target_pos = [0.3, 0.3, 0.5]  # Replace with your target position

# Calculate the joint angles
joint_angles = p.calculateInverseKinematics(robot, end_effector_index, target_pos)

# Set the joint angles
for i in range(num_joints):
    p.setJointMotorControl2(bodyUniqueId=robot,
                            jointIndex=i,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=joint_angles[i])

# Step the simulation forward
for _ in range(10000):  # Run the simulation for 10000 steps
    p.stepSimulation()
    time.sleep(1 / 240.)  # The simulation runs at 240 steps per second

# # Define the joint index
# joint_index = 0  # Change this to the index of the joint you want to control

# # Define the amplitude and frequency of the oscillation
# amplitude = math.pi/2  # The joint will move +/- 45 degrees
# frequency = 2  # The joint will complete a full oscillation in 2 seconds

# # Start the simulation
# for i in range(10000):
#     # Calculate the desired joint position
#     joint_position = amplitude * math.sin(2 * math.pi * frequency * i / 240.)
#
#     # Set the joint position
#     p.setJointMotorControl2(robot, joint_index, p.POSITION_CONTROL, joint_position)
#
#     # Step the simulation forward
#     p.stepSimulation()
#
#     # Wait a bit to create a real-time simulation
#     time.sleep(1 / 240.)  # The simulation runs at 240 steps per second