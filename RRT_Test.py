import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, width, height, depth):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.width = width
        self.height = height
        self.depth = depth
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]
        self.kdtree = KDTree([[self.start.x, self.start.y, self.start.z]])

    def plan(self, max_iter=200):
        for i in range(max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.extend(nearest_node, rnd_node)

            if self.check_collision(new_node, nearest_node):
                self.node_list.append(new_node)
                self.kdtree = KDTree([[node.x, node.y, node.z] for node in self.node_list])

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y, self.node_list[-1].z) <= 1:
                return self.generate_final_course(len(self.node_list) - 1)

        return None  # Cannot find path

    def extend(self, from_node, to_node, length=float('inf')):
        d, theta, phi = self.calc_distance_and_angle(from_node, to_node)
        distance = min(d, length)

        new_node = Node(from_node.x + distance * np.sin(theta) * np.cos(phi),
                        from_node.y + distance * np.sin(theta) * np.sin(phi),
                        from_node.z + distance * np.cos(theta))
        new_node.parent = from_node

        return new_node

    def get_random_node(self):
        rnd = Node(np.random.uniform(0, self.width), np.random.uniform(0, self.height), np.random.uniform(0, self.depth))
        return rnd

    def get_nearest_node(self, node):
        dist, ind = self.kdtree.query([[node.x, node.y, node.z]], k=1)
        return self.node_list[ind[0][0]]

    def check_collision(self, node, parent_node):
        for (ox, oy, oz, size) in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            dz = oz - node.z
            d = dx * dx + dy * dy + dz * dz
            if d <= size**2:
                return False  # collision

        return True  # safe

    def calc_dist_to_goal(self, x, y, z):
        dx = x - self.goal.x
        dy = y - self.goal.y
        dz = z - self.goal.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        theta = np.arccos(dz / d)
        phi = np.arctan2(dy, dx)
        return d, theta, phi

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y, self.goal.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])

        return path