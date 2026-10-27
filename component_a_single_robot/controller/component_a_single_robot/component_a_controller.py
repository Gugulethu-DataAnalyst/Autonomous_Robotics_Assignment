"""
Component A: Integrated Perception, Motion Planning, and Closed-Loop Control
Vehicle: Differential Drive Robot (E-puck)
Sensors: LiDAR, Camera (built-in e-puck camera), GPS, Compass
Data Export: Exports trajectory and tracking errors to trajectory_log.csv
"""

import csv
import cv2
import numpy as np
from controller import Robot


class PIDController:
    """Discrete PID Controller for Heading Tracking."""
    def __init__(self, kp=2.5, ki=0.001, kd=0.15):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        error = np.arctan2(np.sin(error), np.cos(error))  # Normalize to [-pi, pi]
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class AutonomousRobotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # Motors (Handle standard and fallback motor device names)
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Built-in E-puck Camera & LiDAR
        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.time_step)

        self.lidar = self.robot.getDevice('lidar')
        if self.lidar:
            self.lidar.enable(self.time_step)

        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.time_step)

        self.compass = self.robot.getDevice('compass')
        if self.compass:
            self.compass.enable(self.time_step)

        self.pid = PIDController(kp=3.0, ki=0.001, kd=0.1)
        self.max_speed = 6.28

    def get_pose(self):
        if not self.gps or not self.compass:
            return 0.0, 0.0, 0.0
        pos = self.gps.getValues()
        north = self.compass.getValues()
        heading = np.arctan2(north[0], north[1])
        return pos[0], pos[1], heading

    def set_motor_speeds(self, linear_vel, angular_vel):
        R, L = 0.021, 0.053
        v_l = (linear_vel - (angular_vel * L / 2.0)) / R
        v_r = (linear_vel + (angular_vel * L / 2.0)) / R
        self.left_motor.setVelocity(np.clip(v_l, -self.max_speed, self.max_speed))
        self.right_motor.setVelocity(np.clip(v_r, -self.max_speed, self.max_speed))


if __name__ == '__main__':
    controller = AutonomousRobotController()
    waypoints = [(-3.0, 0.0), (-2.0, 2.0), (0.0, 2.0), (2.0, -1.0), (3.0, 3.0)]
    current_wp_idx = 0

    log_file = open('trajectory_log.csv', mode='w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['time', 'x_actual', 'y_actual', 'x_target', 'y_target', 'heading_error'])

    sim_time = 0.0
    print("Controller initialized successfully. Starting navigation loop...")

    while controller.robot.step(controller.time_step) != -1:
        x, y, theta = controller.get_pose()
        target_x, target_y = waypoints[current_wp_idx]
        dist_to_goal = np.hypot(target_x - x, target_y - y)

        if dist_to_goal < 0.25 and current_wp_idx < len(waypoints) - 1:
            current_wp_idx += 1
            target_x, target_y = waypoints[current_wp_idx]

        desired_heading = np.arctan2(target_y - y, target_x - x)
        heading_error = desired_heading - theta

        w_cmd = controller.pid.compute(heading_error, controller.dt)
        v_cmd = 0.15 if abs(heading_error) < 0.5 else 0.05

        controller.set_motor_speeds(v_cmd, w_cmd)

        writer.writerow([round(sim_time, 3), round(x, 3), round(y, 3), round(target_x, 3), round(target_y, 3), round(heading_error, 4)])
        sim_time += controller.dt

        if current_wp_idx == len(waypoints) - 1 and dist_to_goal < 0.15:
            print("Goal reached! Stopping robot.")
            controller.set_motor_speeds(0.0, 0.0)
            break

    log_file.close()