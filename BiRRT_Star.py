import numpy as np
from scipy.spatial import KDTree


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.cost = 0
        self.parent = None
class BiRRTStar:
    def __init__(self, start, goal, obstacle_list, width, height, depth):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.width = width
        self.height = height
        self.depth = depth
        self.obstacle_list = obstacle_list
        self.start_tree = [self.start]
        self.goal_tree = [self.goal]
        self.start_kdtree = KDTree([[self.start.x, self.start.y, self.start.z]])
        self.goal_kdtree = KDTree([[self.goal.x, self.goal.y, self.goal.z]])

    def plan(self, max_iter=200): #duplicate part based on the RRTtest
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

                    if self.calc_dist_to_goal(new_node, nearest_node_goal_tree) <= 1:   #check the distance to the goal
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

                    if self.calc_dist_to_goal(new_node, nearest_node_start_tree) <= 1:
                        return self.generate_final_course(nearest_node_start_tree, new_node)
        return None  # Cannot find path

    def extend(self, from_node, to_node, length=float('inf')):
        d, theta, phi = self.calc_distance_and_angle(from_node, to_node)
        distance = min(d, length)

        new_node = Node(from_node.x + distance * np.sin(theta) * np.cos(phi),
                        from_node.y + distance * np.sin(theta) * np.sin(phi),
                        from_node.z + distance * np.cos(theta))
        new_node.parent = from_node

        return new_node

    def rewire(self, tree, new_node, kdtree):
        radius = 10.0
        neighbor_indices = kdtree.query_ball_point([new_node.x, new_node.y, new_node.z], radius) #find the neighbors within the radius
        for i in neighbor_indices:
            neighbor_node = tree[i]
            if neighbor_node != new_node.parent:
                d, theta, phi = self.calc_distance_and_angle(neighbor_node, new_node) #calculate the distance, theta and phi are the angle and not used in this function
                if new_node.cost + d < neighbor_node.cost:
                    neighbor_node.parent = new_node
                    neighbor_node.cost = new_node.cost + d

    def get_random_node(self):
        rnd = Node(np.random.uniform(0, self.width), np.random.uniform(0, self.height), np.random.uniform(0, self.depth))
        return rnd

    def get_nearest_node(self, node, tree, kdtree):
        dist, ind = kdtree.query([[node.x, node.y, node.z]], k=1)
        return tree[ind[0][0]]

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

    def generate_final_course(self, start_node, goal_node):
        path = [[self.goal.x, self.goal.y, self.goal.z]]
        node = goal_node
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([self.start.x, self.start.y, self.start.z])
        node = start_node.parent
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])

        return path