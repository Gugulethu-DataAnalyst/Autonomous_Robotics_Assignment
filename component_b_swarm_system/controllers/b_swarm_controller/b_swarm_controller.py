"""
Component B: Decentralized Multi-Robot Swarm Control (Reynolds Flocking)
Behaviors: Separation, Alignment, Cohesion, Target Migration
Communication: Webots Emitter/Receiver (V2V Mesh Channel 1) via JSON Strings
Data Export: Telemetry saved to swarm_telemetry.csv by Robot 0
"""

import csv
import json
import numpy as np
from controller import Robot


class ReynoldsSwarmController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # Extract robot ID number from name "epuck_X"
        self.robot_name = self.robot.getName()
        self.robot_id = int(self.robot_name.split('_')[1])

        # Wheel Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Sensors & Comms
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.time_step)

        self.compass = self.robot.getDevice('compass')
        if self.compass:
            self.compass.enable(self.time_step)

        self.emitter = self.robot.getDevice('emitter')
        self.receiver = self.robot.getDevice('receiver')
        if self.receiver:
            self.receiver.enable(self.time_step)

        # Reynolds Weights & Parameters
        self.w_sep = 1.8   # Separation weight
        self.w_align = 1.0 # Alignment weight
        self.w_coh = 1.2   # Cohesion weight
        self.w_mig = 1.5   # Migration weight
        self.d_safe = 0.4  # Separation threshold (meters)
        self.max_speed = 6.28

    def get_pose(self):
        pos = self.gps.getValues()
        north = self.compass.getValues()
        heading = np.arctan2(north[0], north[1])
        return np.array([pos[0], pos[1]]), heading

    def broadcast_state(self, pos, heading):
        """Broadcasts state as a JSON string to all neighbors over V2V mesh."""
        data = {
            'id': self.robot_id,
            'x': float(pos[0]),
            'y': float(pos[1]),
            'heading': float(heading)
        }
        message = json.dumps(data)
        self.emitter.send(message.encode('utf-8'))

    def receive_neighbor_states(self):
        """Reads incoming JSON string telemetry from nearby swarm mates."""
        neighbors = []
        while self.receiver.getQueueLength() > 0:
            if hasattr(self.receiver, 'getString'):
                packet = self.receiver.getString()
            else:
                packet = self.receiver.getData().decode('utf-8')

            try:
                data = json.loads(packet)
                if data['id'] != self.robot_id:
                    neighbors.append({
                        'id': data['id'],
                        'pos': np.array([data['x'], data['y']]),
                        'heading': data['heading']
                    })
            except Exception:
                pass

            self.receiver.nextPacket()
        return neighbors

    def compute_flocking_velocity(self, pos, heading, neighbors, target_pos):
        v_sep = np.array([0.0, 0.0])
        v_align = np.array([0.0, 0.0])
        v_coh = np.array([0.0, 0.0])

        if len(neighbors) > 0:
            neighbor_positions = []
            neighbor_headings = []

            for n in neighbors:
                dist = np.linalg.norm(pos - n['pos'])
                neighbor_positions.append(n['pos'])
                neighbor_headings.append(n['heading'])

                # 1. Separation Rule
                if dist < self.d_safe and dist > 0:
                    v_sep += (pos - n['pos']) / (dist ** 2)

            # 2. Cohesion Rule
            center_of_mass = np.mean(neighbor_positions, axis=0)
            v_coh = center_of_mass - pos

            # 3. Alignment Rule
            mean_heading = np.mean(neighbor_headings)
            v_align = np.array([np.cos(mean_heading), np.sin(mean_heading)])

        # 4. Migration Rule (Target Attraction)
        v_mig = target_pos - pos
        if np.linalg.norm(v_mig) > 0:
            v_mig = v_mig / np.linalg.norm(v_mig)

        # Combine Vector Rules
        desired_vec = (self.w_sep * v_sep) + (self.w_align * v_align) + \
                      (self.w_coh * v_coh) + (self.w_mig * v_mig)

        desired_heading = np.arctan2(desired_vec[1], desired_vec[0])
        heading_error = desired_heading - heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Differential Drive Kinematics
        w_cmd = 3.0 * heading_error
        v_cmd = 0.12 if abs(heading_error) < 0.6 else 0.03

        R, L = 0.021, 0.053
        v_l = (v_cmd - (w_cmd * L / 2.0)) / R
        v_r = (v_cmd + (w_cmd * L / 2.0)) / R
        return np.clip(v_l, -self.max_speed, self.max_speed), np.clip(v_r, -self.max_speed, self.max_speed)


if __name__ == '__main__':
    controller = ReynoldsSwarmController()
    target_goal = np.array([3.5, 3.5])

    # Robot 0 manages the CSV telemetry logger
    log_file = None
    writer = None
    if controller.robot_id == 0:
        log_file = open('swarm_telemetry.csv', mode='w', newline='')
        writer = csv.writer(log_file)
        writer.writerow(['time', 'robot_id', 'x', 'y', 'heading'])

    sim_time = 0.0
    print(f"Swarm Robot {controller.robot_id} active.")

    while controller.robot.step(controller.time_step) != -1:
        pos, heading = controller.get_pose()
        controller.broadcast_state(pos, heading)
        neighbors = controller.receive_neighbor_states()

        v_l, v_r = controller.compute_flocking_velocity(pos, heading, neighbors, target_goal)
        controller.left_motor.setVelocity(v_l)
        controller.right_motor.setVelocity(v_r)

        # Log Robot 0 pose & neighbors for analysis
        if controller.robot_id == 0 and writer:
            writer.writerow([round(sim_time, 3), controller.robot_id, round(pos[0], 3), round(pos[1], 3), round(heading, 4)])
            for n in neighbors:
                writer.writerow([round(sim_time, 3), n['id'], round(n['pos'][0], 3), round(n['pos'][1], 3), round(n['heading'], 4)])
            sim_time += controller.dt

        # Stop when close to global target
        if np.linalg.norm(target_goal - pos) < 0.3:
            controller.left_motor.setVelocity(0.0)
            controller.right_motor.setVelocity(0.0)
            break

    if log_file:
        log_file.close()