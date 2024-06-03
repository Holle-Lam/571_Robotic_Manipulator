from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p
import time


ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')


# Define the active links mask. Set the value for 'EEF_Joint' to False.
active_links_mask = [False, True, True, True, True]

# Load the URDF
chain = Chain.from_urdf_file("571_robotic_arm_2.0/urdf/571_robotic_arm_2.0.urdf", active_links_mask=active_links_mask)

# Define the zero position
zero_position = [0] * len(active_links_mask)

# Compute the forward kinematics for the zero position
end_effector_position_zero = chain.forward_kinematics(zero_position)

print("End effector position at zero position: ", end_effector_position_zero)


# # Define the target position
# target_position = [-0.1, 0.2, 0.2]
#
# # Compute the inverse kinematics
# joint_angles = chain.inverse_kinematics(target_position)
#
# # Compute the forward kinematics
# end_effector_position = chain.forward_kinematics(joint_angles)
base_link_offset = [-0.0191416107228431, 0.0401313314432728, 0.00504410822541169] #the offset of the base link
# Define the target position with the offset
joint_angles = chain.inverse_kinematics([-0.14 + base_link_offset[0], -0.14 + base_link_offset[1], 0.15 + base_link_offset[2]])
print("Commanded position")
print([-0.1 + base_link_offset[0], -0.1 + base_link_offset[1], 0.2 + base_link_offset[2]])
# # Compute the forward kinematics
end_effector_position = chain.forward_kinematics(joint_angles)

chain.plot(joint_angles, ax)
# chain.plot(chain.inverse_kinematics([0.2, 0.2, 0.2]), ax)
# Define the zero position

# extract translational coordinates of the end effector
x = end_effector_position[0, 3] - base_link_offset[0]
y = end_effector_position[1, 3] - base_link_offset[1]
z = end_effector_position[2, 3] - base_link_offset[2]

print("Computed joint angles for inverse kinematics: ", joint_angles)
print("End effector position computed from forward kinematics: ", [x, y, z])


# # Connect to the physics server
# p.connect(p.GUI)
#
# # Load the URDF file
# robot = p.loadURDF("571_robotic_arm_2.0/urdf/571_robotic_arm_2.0.urdf")
#
# # Create a fixed constraint between the robot base and world
# p.createConstraint(parentBodyUniqueId=robot,
#                    parentLinkIndex=-1,  # -1 indicates the base
#                    childBodyUniqueId=-1,  # -1 indicates the world
#                    childLinkIndex=-1,
#                    jointType=p.JOINT_FIXED,
#                    jointAxis=[0, 0, 0],
#                    parentFramePosition=[0, 0, 0],
#                    childFramePosition=[0, 0, 0])
#
# # Get the number of joints
# num_joints = p.getNumJoints(robot)
#
# # Get the indices of the active joints
# active_joint_indices = [i for i in range(num_joints) if active_links_mask[i]]
#
# joints_to_control = [0, 1, 2, 3]
# # joint_positions = joint_angles[1:5]
# joint_positions = joint_angles[1:5]
# # Set the joint angles
# p.setJointMotorControlArray(robot, joints_to_control, p.POSITION_CONTROL, targetPositions=joint_positions)
#
# # Step the simulation
# for _ in range(1000):
#     p.stepSimulation()
#     time.sleep(0.01)
#
# # Get the index of the end effector (last link)
# end_effector_index = num_joints -1
#
# # Get the state of the end effector
# end_effector_state = p.getLinkState(robot, end_effector_index)
#
# # The position is the first element of the state
# end_effector_position = end_effector_state[0]

# print("End effector position: ", end_effector_position)

matplotlib.pyplot.show()

