import pybullet as p
import pybullet_data
from BiRRT_Star import BiRRTStar, Node  # Assuming BiRRT_Star.py is in the same directory

# Connect to the physics server
p.connect(p.GUI)

# Add the data directory to PyBullet's path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the URDF file for the default manipulator
manipulator_id = p.loadURDF("kuka_iiwa/model.urdf")

# Set the initial positions of the manipulator
p.resetBasePositionAndOrientation(manipulator_id, [0, 0, 0], [0, 0, 0, 1])

# Get the number of joints
num_joints = p.getNumJoints(manipulator_id)

# Save the original joint states
original_joint_states = [p.getJointState(manipulator_id, i)[0] for i in range(num_joints)]

# Set the new joint angles
new_joint_angles = [0.1] * num_joints  # Replace with your new joint angles
p.setJointMotorControlArray(manipulator_id, range(num_joints), p.POSITION_CONTROL, targetPositions=new_joint_angles)

# Step the simulation forward
p.stepSimulation()

# Check for collisions
collisions = p.getClosestPoints(manipulator_id, -1, 0)

# If a collision is detected, then the new joint angles would cause a collision
if collisions:
    print("The new joint angles would cause a collision.")

# Reset the joint angles to their original values
for i in range(num_joints):
    p.resetJointState(manipulator_id, i, original_joint_states[i])

# Define the start and goal points
start = [0, 0, 0]
goal = [1, 1, 1]

# Define the obstacle list
obstacle_list = []  # Add your obstacles here

# Create an instance of the BiRRTStar class
birrt_star = BiRRTStar(start, goal, obstacle_list, width=2, height=2, depth=2)

# Generate the path
path = birrt_star.plan()

# Disconnect from the physics server
p.disconnect()

# Print the path
if path is not None:
    print("Path found:")
    for point in path:
        print(point)
else:
    print("No path found.")