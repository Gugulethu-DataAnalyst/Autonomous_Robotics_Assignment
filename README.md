## The Autonomous Systems & Robotics Assignment
**Course Module:** Autonomous Systems and Robotics 700  
**Simulation Platform:** Webots R2025a (Python API)  

--------------------------------------

## The Overview
This repository contains the implementation and experimention of autonomous mobile robot systems across two components:

1. ## Component A: A Single Robot Path Planning & Perception**
   1.1 **Environment:** $10\text{m} \times 10\text{m}$ Webots arena (`new_world.wbt`) with static box obstacles.
   1.2 **Sensors:** LiDAR, RGB Camera, GPS, Compass.
   1.3 **Algorithmic Core:** 2D Occupancy Grid Mapping, Global Pathfinding, and Closed-Loop Differential Drive PID Heading Tracking.

2. **Component B: Decentralized Multi-Robot Swarm Control**
   2.1 **Environment:** $10\text{m} \times 10\text{m}$ open arena (`component_b_world.wbt`).
   2.2 **Communication:** V2V Mesh Networking via Webots Emitter & Receiver (JSON payload messaging).
   2.3 **Algorithmic Core:** Decentralized Reynolds Flocking Model incorporating Separation, Alignment, Cohesion, and Target Attraction.

---

## Structure
`Component_A_Single_Robot/`: The simulation world, $A^*$ controller, and telemetry analysis notebook.
- `Component_B_Swarm_System/`: Multi-robot world, Reynolds flocking controller, and a swarm order analysis notebook.

---

## Execution
1. You have to open desired world file in Webots (`new_world.wbt` or `component_b_world.wbt`).
2. Run simulation to output trajectory CSV logs.
3. Open corresponding Jupyter notebook in `notebooks/` to process telemetry and display publication graphics.