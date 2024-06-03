import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import pybullet as p
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
#a class for the 

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.cost = 0
        self.parent = None
class BiRRTStar:
    def __init__(self, start, goal, obstacle_list):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.chain = Chain.from_urdf_file("571_robotic_arm_2.0/urdf/571_robotic_arm_2.0.urdf", active_links_mask=[False, True, True, True, True])
        self.base_link_offset = [-0.0191416107228431, 0.0401313314432728,
                                 0.00504410822541169]  # the offset of the base link
        # throw exception if the start and goal are not valid
        if not self.valid_position(self.start):
            raise ValueError("Invalid start position")

        if not self.valid_position(self.goal):
            raise ValueError("Invalid goal position")

        self.obstacle_list = obstacle_list
        self.start_tree = [self.start]
        self.goal_tree = [self.goal]
        self.start_kdtree = KDTree([[self.start.x, self.start.y, self.start.z]])
        self.goal_kdtree = KDTree([[self.goal.x, self.goal.y, self.goal.z]])
        self.extend_length = 0.01

    def plan(self, max_iter=500): #duplicate part based on the RRTtest
        for i in range(max_iter):
            if i%2==0: #start direction -> goal direction
                rnd_node = self.get_random_node()
                nearest_node = self.get_nearest_node(rnd_node, self.start_tree, self.start_kdtree)
                new_node = self.extend(nearest_node, rnd_node)

                if self.check_collision(new_node, nearest_node):
                    self.start_tree.append(new_node)
                    self.start_kdtree = KDTree([[node.x, node.y, node.z] for node in self.start_tree]) #add the kdtree start direction -> goal direction
                    self.rewire(self.start_tree, new_node, self.start_kdtree)   #rewire the tree
                    nearest_node_goal_tree = self.get_nearest_node(new_node, self.goal_tree, self.goal_kdtree)

                    if self.calc_distance_between_nodes(new_node, nearest_node_goal_tree) <= self.extend_length:   #check the distance to the goal
                        return self.generate_final_course(new_node, nearest_node_goal_tree)
            else: #goal direction -> start direction
                rnd_node = self.get_random_node()
                nearest_node = self.get_nearest_node(rnd_node, self.goal_tree, self.goal_kdtree)
                new_node = self.extend(nearest_node, rnd_node)

                if self.check_collision(new_node, nearest_node):
                    self.goal_tree.append(new_node)
                    self.goal_kdtree = KDTree([[node.x, node.y, node.z] for node in self.goal_tree]) #add the kdtree goal direction -> start direction
                    self.rewire(self.goal_tree, new_node, self.goal_kdtree)   #rewire the tree  
                    nearest_node_start_tree = self.get_nearest_node(new_node, self.start_tree, self.start_kdtree)

                    if self.calc_distance_between_nodes(new_node, nearest_node_start_tree) <= self.extend_length:   #check the distance to the start
                        return self.generate_final_course(nearest_node_start_tree, new_node)
        return None  # Cannot find path

    def extend(self, from_node, to_node):
        length = self.extend_length
        d, theta, phi = self.calc_distance_and_angle(from_node, to_node)
        distance = min(d, length)

        new_node = Node(from_node.x + distance * np.sin(theta) * np.cos(phi),
                        from_node.y + distance * np.sin(theta) * np.sin(phi),
                        from_node.z + distance * np.cos(theta))
        new_node.parent = from_node
        new_node.cost = from_node.cost + distance  # Set the cost of the new node

        return new_node

    def rewire(self, tree, new_node, kdtree):
        radius = .05
        neighbor_indices = kdtree.query_ball_point([new_node.x, new_node.y, new_node.z], radius) #find the neighbors within the radius
        for i in neighbor_indices:
            neighbor_node = tree[i]
            if neighbor_node != new_node.parent:
                d, theta, phi = self.calc_distance_and_angle(neighbor_node, new_node) #calculate the distance, theta and phi are the angle and not used in this function
                if new_node.cost + d < neighbor_node.cost and new_node!=neighbor_node:
                    neighbor_node.parent = new_node
                    neighbor_node.cost = new_node.cost + d

    def get_random_node(self):
        """Generates a random node placed within the bounding box of the environment."""
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounding_box()
        rnd = Node(np.random.uniform(x_min, x_max),
                   np.random.uniform(y_min, y_max),
                   np.random.uniform(z_min, z_max))
        return rnd

    def get_nearest_node(self, node, tree, kdtree):
        dist, ind = kdtree.query([[node.x, node.y, node.z]], k=1)

        return tree[int(ind)]


    def check_collision(self, node, parent_node):

        if not self.valid_position(node):
            return False # unachievable position

        for (ox, oy, oz, size) in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            dz = oz - node.z
            d = dx * dx + dy * dy + dz * dz
            if d <= size**2:
                return False  # collision

        return True  # safe

    def calc_distance_between_nodes(self, node1, node2):
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        dz = node2.z - node1.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d == 0:
            return d, 0, 0
        theta = np.arccos(dz / d)
        phi = np.arctan2(dy, dx)
        return d, theta, phi

    def generate_final_course(self, start_node, goal_node):
        goal_path = []
        start_path = []

        # iterate through goal tree
        node = goal_node
        while node.parent is not None:
            goal_path.append([node.x, node.y, node.z])
            node = node.parent
        goal_path.append([node.x, node.y, node.z])

        # iterate through start tree
        node = start_node
        while node.parent is not None:
            start_path.append([node.x, node.y, node.z])
            node = node.parent
        start_path.append([node.x, node.y, node.z])

        return start_path[::-1] + goal_path

    def plot_trees(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the start tree
        for node in self.start_tree:
            if node.parent:
                xs = [node.x, node.parent.x]
                ys = [node.y, node.parent.y]
                zs = [node.z, node.parent.z]
                ax.plot(xs, ys, zs, color='blue')

        # Plot the goal tree
        for node in self.goal_tree:
            if node.parent:
                xs = [node.x, node.parent.x]
                ys = [node.y, node.parent.y]
                zs = [node.z, node.parent.z]
                ax.plot(xs, ys, zs, color='green')

        # Plot the final path
        if path is not None:
            xs = [point[0] for point in path]
            ys = [point[1] for point in path]
            zs = [point[2] for point in path]
            ax.plot(xs, ys, zs, color='red')

        ax.scatter([self.start.x, self.goal.x], [self.start.y, self.goal.y], [self.start.z, self.goal.z],
                   color='red')  # Start and goal points
        plt.show()


    def valid_position(self, node2):
        # checks if the move from node1 to node2 is valid kinematically

        # offset the base link so that the base link is at the origin

        # Compute the inverse kinematics
        joint_angles = self.chain.inverse_kinematics([node2.x + self.base_link_offset[0],
                                                      node2.y + self.base_link_offset[1],
                                                      node2.z + self.base_link_offset[2]])
        # Compute the forward kinematics
        end_effector_position = self.chain.forward_kinematics(joint_angles)

        # extract translational coordinates of the end effector
        x = end_effector_position[0, 3] - self.base_link_offset[0]
        y = end_effector_position[1, 3] - self.base_link_offset[1]
        z = end_effector_position[2, 3] - self.base_link_offset[2]

        #calculate distance between new end effector position and node2
        dx = x - node2.x
        dy = y - node2.y
        dz = z - node2.z
        d = np.sqrt(dx * dx + dy * dy + dz * dz)

        return np.all(d < 0.025)  # check if the distance is less than 0.01

    def bounding_box(self):
        """
        This function returns a bounding box that contains enough space to cover the entire robot's workspace.
        """
        # Define the bounding box
        x_min = -0.4
        x_max = 0.4
        y_min = -0.4
        y_max = 0.4
        z_min = 0.0
        z_max = 0.4

        return x_min, x_max, y_min, y_max, z_min, z_max