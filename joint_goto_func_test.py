from robot_controller import robot_controller
import time
import numpy as np



def main():
    RC = robot_controller()
    RC.communication_begin()
    goals = np.ones(RC.joint_num) * 90 # degree
    speeds = np.ones(RC.joint_num) * 5 # degree/s

    RC.joints_homing()
    while True:
        
        time.sleep(3)
        RC.gripper_close()
        goals = -goals
        print("\ngoals: ", goals)
        print("speeds: ", speeds)
        RC.joints_goto(goals, speeds)
        RC.gripper_open()

    RC.communication_end()


if __name__ == "__main__":
    main()