from BiRRT_Star import BiRRTStar, Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p
import pybullet_data
import time

# Define the start and goal points
start = [0.05, 0.1, 0.2]
goal = [-0.1, -0.1, 0.15]  # Very close to the start point

# Define the obstacle list
obstacle_list = []  # No obstacles

# Create an instance of the BiRRTStar class
birrt_star = BiRRTStar(start, goal, obstacle_list)

# Generate the path
path = birrt_star.plan()

# Plot the trees
birrt_star.plot_trees(path)

# Print the path
if path is not None:
    print("Path found:")
    for point in path:
        print(point)
else:
    print("No path found.")

# Plot the path
if path is not None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [point[0] for point in path]
    ys = [point[1] for point in path]
    zs = [point[2] for point in path]
    ax.plot(xs, ys, zs)
    ax.scatter([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], color='red')  # Start and goal points
    plt.show()

# Connect to the physics server
p.connect(p.GUI)

# Set the camera position
p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.0, 0.0, 0.0])

# Load the URDF file
robot = p.loadURDF("571_robotic_arm_2.0/urdf/571_robotic_arm_2.0.urdf")

# Create a fixed constraint between the robot base and world
p.createConstraint(parentBodyUniqueId=robot,
                   parentLinkIndex=-1,  # -1 indicates the base
                   childBodyUniqueId=-1,  # -1 indicates the world
                   childLinkIndex=-1,
                   jointType=p.JOINT_FIXED,
                   jointAxis=[0, 0, 0],
                   parentFramePosition=[0, 0, 0],
                   childFramePosition=[0, 0, 0])

# Get the number of joints of the robot
num_joints = p.getNumJoints(robot)

# Get the indices of the active joints
active_joint_indices = [0, 1, 2, 3]

# Navigate the arm along the path
for point in path:
    # Compute the inverse kinematics for the current point
    joint_angles = birrt_star.chain.inverse_kinematics([point[0] + birrt_star.base_link_offset[0],
                                                        point[1] + birrt_star.base_link_offset[1],
                                                        point[2] + birrt_star.base_link_offset[2]])

    # Set the joint angles
    p.setJointMotorControlArray(robot, active_joint_indices, p.POSITION_CONTROL, targetPositions=joint_angles[1:5])

    # Step the simulation
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)