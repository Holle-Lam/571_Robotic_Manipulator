from robot_controller import robot_controller
import time
import numpy as np



def main():
    RC = robot_controller()
    RC.communication_begin()

    RC.joints_homing()
    while True:
        time.sleep(1)
        # Generate goals array with 4 random floats from -90 to 90
        goals = np.random.randint(-90, 90, size=RC.joint_num)
        
        # Generate speeds array with 4 random floats from 10 to 30
        speeds = np.random.randint(5, 20, size=RC.joint_num)
        print("\ngoals: ", goals)
        print("speeds: ", speeds)
        RC.joints_goto(goals, speeds)

    RC.communication_end()


if __name__ == "__main__":
    main()