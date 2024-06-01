import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version p.GUI for graphical version

# Load the URDF file
robot = p.loadURDF("571robotic_arm_final_version/urdf/571robotic_arm_final_version.urdf")

# Create a fixed constraint between the robot base and world
p.createConstraint(parentBodyUniqueId=robot,
                   parentLinkIndex=-1,  # -1 indicates the base
                   childBodyUniqueId=-1,  # -1 indicates the world
                   childLinkIndex=-1,
                   jointType=p.JOINT_FIXED,
                   jointAxis=[0, 0, 0],
                   parentFramePosition=[0, 0, 0],
                   childFramePosition=[0, 0, 0])

# Get the number of joints
num_joints = p.getNumJoints(robot)

# Define the range of joint angles
joint_angle_ranges = np.linspace(-np.pi/2, np.pi/2, 10)  # Replace with your joint angle ranges

# Initialize a list to store the end effector positions
end_effector_positions = []

# Iterate over all combinations of joint angles
for joint_angles in np.nditer(np.meshgrid(*[joint_angle_ranges]*num_joints)):
    # Set the joint angles
    for i in range(num_joints):
        p.setJointMotorControl2(bodyUniqueId=robot,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_angles[i])

    # Step the simulation forward
    p.stepSimulation()

    # Get the new state of the end effector
    end_effector_state = p.getLinkState(robot, num_joints - 1)

    # The new position of the end effector in world coordinates
    new_end_effector_pos = end_effector_state[0]

    # Store the end effector position
    end_effector_positions.append(new_end_effector_pos)

# Convert the list of end effector positions to a numpy array
end_effector_positions = np.array(end_effector_positions)

# Monte Carlo simulation
num_samples = 10000  # Number of samples for the Monte Carlo simulation
valid_positions = []  # List to store the valid positions
invalid_positions = []  # List to store the invalid positions

for _ in range(num_samples):
    # Generate a random end effector position within the working range of the arm
    random_pos = np.random.uniform(np.min(end_effector_positions, axis=0), np.max(end_effector_positions, axis=0))

    # Calculate the joint angles for the random position
    joint_angles = p.calculateInverseKinematics(robot, num_joints - 1, random_pos)

    # Check if the joint angles are within the joint limits
    joint_info = [p.getJointInfo(robot, i) for i in range(num_joints)]
    joint_limits = [(info[8], info[9]) for info in joint_info]  # Lower and upper joint limits

    if all(lower <= angle <= upper for angle, (lower, upper) in zip(joint_angles, joint_limits)):
        valid_positions.append(random_pos)
    else:
        invalid_positions.append(random_pos)

# Convert the lists of valid and invalid positions to numpy arrays
valid_positions = np.array(valid_positions)
invalid_positions = np.array(invalid_positions)

# Print the valid and invalid positions
print(f"Valid positions: {valid_positions}")
print(f"Invalid positions: {invalid_positions}")

# Create a new figure
fig = plt.figure()

# Create a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot of the valid positions
ax.scatter(valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2], c='b', marker='o', label='Valid positions')

# Create a scatter plot of the invalid positions
ax.scatter(invalid_positions[:, 0], invalid_positions[:, 1], invalid_positions[:, 2], c='r', marker='x', label='Invalid positions')

# Set the plot title and labels
ax.set_title('Valid and Invalid Positions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a legend
ax.legend()

# Show the plot
plt.show()

# Convert the valid positions to a DataFrame
df_valid_positions = pd.DataFrame(valid_positions, columns=['X', 'Y', 'Z'])

# Save the DataFrame to a CSV file
df_valid_positions.to_csv('valid_positions.csv', index=False)

# # Load the valid positions from the CSV file
# df_valid_positions = pd.read_csv('valid_positions.csv')
#
# # Convert the DataFrame to a numpy array
# valid_positions = df_valid_positions.values

def is_point_valid(point, valid_positions):
    """
    Check if a point is valid based on the Monte Carlo simulation data.

    Parameters:
    point: The point to check.
    valid_positions: The valid positions obtained from the Monte Carlo simulation.

    Returns:
    True if the point is valid, False otherwise.
    """
    # Calculate the distance between the point and each valid position
    distances = np.linalg.norm(valid_positions - point, axis=1)

    # If the minimum distance is less than a threshold, the point is considered valid
    return np.min(distances) < 0.1  # Replace 0.1 with your threshold