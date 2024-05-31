import pybullet as p

# Load the URDF file for the manipulator
manipulator_id = p.loadURDF("manipulator.urdf")

# Set the initial positions of the manipulator
p.resetBasePositionAndOrientation(manipulator_id, [0, 0, 0], [0, 0, 0, 1])

# Get the number of joints
num_joints = p.getNumJoints(manipulator_id)

# Define the target position for the end-effector
target_position = [0.5, 0.5, 0.5]  # Replace with your target position

# Calculate the inverse kinematics
joint_angles = p.calculateInverseKinematics(manipulator_id, num_joints - 1, target_position)

# Set the joint angles
p.setJointMotorControlArray(manipulator_id, range(num_joints), p.POSITION_CONTROL, targetPositions=joint_angles)

# Step the simulation forward
p.stepSimulation()