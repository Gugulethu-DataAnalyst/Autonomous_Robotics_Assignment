"""
Component A: Full Perception, A* Path Planning, and PID Closed-Loop Control
Vehicle: Differential Drive Robot (E-puck)
Sensors: LiDAR, Camera, GPS, Compass
Data Export: Saves trajectory metrics and tracking performance to trajectory_log.csv
"""

import csv
import heapq
import cv2
import numpy as np
from controller import Robot


class AStarPlanner:
    """A* Global Path Planner for 2D Occupancy Grid Navigation."""
    def __init__(self, grid_size=(50, 50), resolution=0.2):
        self.grid_size = grid_size
        self.resolution = resolution
        self.origin_offset = np.array([grid_size[0] / 2.0, grid_size[1] / 2.0])

    def world_to_grid(self, wx, wy):
        """Converts world coordinates (meters) to grid indices."""
        gx = int(wx / self.resolution + self.origin_offset[0])
        gy = int(wy / self.resolution + self.origin_offset[1])
        return (np.clip(gx, 0, self.grid_size[0] - 1), np.clip(gy, 0, self.grid_size[1] - 1))

    def grid_to_world(self, gx, gy):
        """Converts grid indices to world coordinates (meters)."""
        wx = (gx - self.origin_offset[0]) * self.resolution
        wy = (gy - self.origin_offset[1]) * self.resolution
        return (wx, wy)

    def heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return np.hypot(b[0] - a[0], b[1] - a[1])

    def plan(self, start_w, goal_w, grid):
        start = self.world_to_grid(*start_w)
        goal = self.world_to_grid(*goal_w)

        # 8-connected movement directions
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                path.append(self.grid_to_world(*start))
                return path[::-1]  # Return path from start to goal

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < self.grid_size[0] and 0 <= neighbor[1] < self.grid_size[1]:
                    if grid[neighbor[0], neighbor[1]] == 1:
                        continue  # Skip occupied cells

                    move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    tentative_g = g_score[current] + move_cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("A* Warning: No path found! Falling back to direct goal line.")
        return [start_w, goal_w]


class PIDController:
    """Discrete PID Controller for Heading Tracking."""
    def __init__(self, kp=2.2, ki=0.0, kd=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        error = np.arctan2(np.sin(error), np.cos(error))
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class AutonomousRobotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Sensors
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

        self.pid = PIDController(kp=2.2, ki=0.0, kd=0.2)
        self.planner = AStarPlanner(grid_size=(50, 50), resolution=0.2)
        self.max_speed = 6.28

    def get_pose(self):
        if not self.gps or not self.compass:
            return 0.0, 0.0, 0.0
        pos = self.gps.getValues()
        north = self.compass.getValues()
        heading = np.arctan2(north[0], north[1])
        return pos[0], pos[1], heading

    def build_known_obstacle_grid(self, grid_size=(50, 50), resolution=0.2):
        """Builds occupancy grid incorporating arena obstacles."""
        grid = np.zeros(grid_size, dtype=int)
        # Bounding boxes for obs_1 (-2.0, -1.0) and obs_2 (1.0, 1.5)
        for gx in range(grid_size[0]):
            for gy in range(grid_size[1]):
                wx, wy = self.planner.grid_to_world(gx, gy)
                # obs_1 inflation box
                if (-3.7 <= wx <= -0.3) and (-1.5 <= wy <= -0.5):
                    grid[gx, gy] = 1
                # obs_2 inflation box
                if (0.5 <= wx <= 1.5) and (0.0 <= wy <= 3.0):
                    grid[gx, gy] = 1
        return grid

    def set_motor_speeds(self, linear_vel, angular_vel):
        R, L = 0.021, 0.053
        v_l = (linear_vel - (angular_vel * L / 2.0)) / R
        v_r = (linear_vel + (angular_vel * L / 2.0)) / R
        self.left_motor.setVelocity(np.clip(v_l, -self.max_speed, self.max_speed))
        self.right_motor.setVelocity(np.clip(v_r, -self.max_speed, self.max_speed))


if __name__ == '__main__':
    controller = AutonomousRobotController()
    
    # 1. Step simulation once to initialize sensor buffers
    controller.robot.step(controller.time_step)
    start_pose = controller.get_pose()[:2]
    goal_pose = (3.0, 3.0)

    # 2. Build Occupancy Grid & Generate A* Optimal Path
    occupancy_grid = controller.build_known_obstacle_grid()
    waypoints = controller.planner.plan(start_pose, goal_pose, occupancy_grid)
    print(f"A* Path Planning Complete: Generated {len(waypoints)} waypoints.")

    current_wp_idx = 0
    log_file = open('trajectory_log.csv', mode='w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['time', 'x_actual', 'y_actual', 'x_target', 'y_target', 'heading_error'])

    sim_time = 0.0

    while controller.robot.step(controller.time_step) != -1:
        x, y, theta = controller.get_pose()
        target_x, target_y = waypoints[current_wp_idx]
        dist_to_goal = np.hypot(target_x - x, target_y - y)

        if dist_to_goal < 0.20 and current_wp_idx < len(waypoints) - 1:
            current_wp_idx += 1
            target_x, target_y = waypoints[current_wp_idx]

        desired_heading = np.arctan2(target_y - y, target_x - x)
        heading_error = desired_heading - theta

        w_cmd = controller.pid.compute(heading_error, controller.dt)
        v_cmd = 0.15 if abs(heading_error) < 0.4 else 0.05

        controller.set_motor_speeds(v_cmd, w_cmd)

        writer.writerow([round(sim_time, 3), round(x, 3), round(y, 3), round(target_x, 3), round(target_y, 3), round(heading_error, 4)])
        sim_time += controller.dt

        if current_wp_idx == len(waypoints) - 1 and dist_to_goal < 0.15:
            print("A* Target Goal Reached! Stopping robot.")
            controller.set_motor_speeds(0.0, 0.0)
            break

    log_file.close()