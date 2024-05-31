import pybullet as p
import pybullet_data
import numpy as np
import time

# Connect to PyBullet
physicsClient = p.connect(p.GUI)

# Add the URDF file path to PyBullet's search path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the URDF
robot = p.loadURDF("571_robotic_arm_urdf_mesh/urdf/571_robotic_arm_urdf_mesh.urdf")

# Get the total number of joints
num_joints = p.getNumJoints(robot)

# Define a set of actions
actions = np.linspace(-np.pi, np.pi, 100)  # Replace with your own actions

# Initialize a list to store the data
data = []

# Run the simulation
for action in actions:
    for i in range(num_joints):
        # Get the initial state of the robot and the end effector
        initial_robot_state = p.getJointState(robot, i)
        initial_end_effector_state = p.getLinkState(robot, num_joints - 1)

        # Set the joint angle
        p.setJointMotorControl2(bodyUniqueId=robot,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=action)

        # Step the simulation forward
        p.stepSimulation()
        time.sleep(1 / 240.)  # The simulation runs at 240 steps per second

        # Get the final state of the robot and the end effector
        final_robot_state = p.getJointState(robot, i)
        final_end_effector_state = p.getLinkState(robot, num_joints - 1)

        # Store the data
        data.append((initial_robot_state, initial_end_effector_state, action, final_robot_state, final_end_effector_state))

# At this point, `data` is a list of tuples, where each tuple contains:
# - The initial state of the robot
# - The initial state of the end effector
# - The action taken
# - The final state of the robot
# - The final state of the end effector