from BiRRT_Star import BiRRTStar, Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the start and goal points
start = [0, 0, 0]
goal = [20, 20, 20]  # Very close to the start point

# Define the obstacle list
obstacle_list = []  # No obstacles

# Create an instance of the BiRRTStar class
birrt_star = BiRRTStar(start, goal, obstacle_list, width=20, height=20, depth=20)

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

