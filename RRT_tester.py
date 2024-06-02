from BiRRT_Star import BiRRTStar, Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the start and goal points
start = [0, 0, 0]
goal = [0.1, 0.1, 0.1]  # Very close to the start point

# Define the obstacle list
obstacle_list = []  # No obstacles

# Create an instance of the BiRRTStar class
birrt_star = BiRRTStar(start, goal, obstacle_list, width=2, height=2, depth=2)

# Generate the path
path = birrt_star.plan()

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