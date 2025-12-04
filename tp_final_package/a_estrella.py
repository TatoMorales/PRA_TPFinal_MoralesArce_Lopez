import math
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan


# Reutilizado de planning_framework.py: funciones auxiliares del algoritmo A*
def get_neighborhood(cell: Tuple[int, int], occ_map_shape: Tuple[int, int]) -> List[List[int]]:
    neighbors: List[List[int]] = []
    x, y = cell
    nx, ny = occ_map_shape

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx_ = x + dx
            ny_ = y + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny:
                neighbors.append([nx_, ny_])

    return neighbors


def get_edge_cost(parent: Tuple[int, int], child: Tuple[int, int], occ_map: np.ndarray) -> float:
    px, py = child
    p_occ = occ_map[px, py]

    if p_occ >= 1.0:
        return np.inf

    dx = parent[0] - child[0]
    dy = parent[1] - child[1]

    if dx == 0 or dy == 0:
        base_cost = 1.0
    else:
        base_cost = math.sqrt(2.0)

    return base_cost


def get_heuristic(cell: Tuple[int, int], goal: Tuple[int, int]) -> float:
    dx = float(cell[0] - goal[0])
    dy = float(cell[1] - goal[1])
    return math.sqrt(dx * dx + dy * dy)


class AEstrellaPlanner:
    """Versión sin ROS del framework de planificación para reutilizar dentro del nodo."""

    def plan(self, occ_map: np.ndarray, start: np.ndarray, goal: np.ndarray):
        costs = np.ones(occ_map.shape) * np.inf
        closed_flags = np.zeros(occ_map.shape)
        predecessors = -np.ones(occ_map.shape + (2,), dtype=int)

        heuristic = np.zeros(occ_map.shape)
        for x in range(occ_map.shape[0]):
            for y in range(occ_map.shape[1]):
                heuristic[x, y] = get_heuristic([x, y], goal)

        parent = start
        costs[start[0], start[1]] = 0

        while not np.array_equal(parent, goal):
            open_costs = np.where(closed_flags == 1, np.inf, costs) + heuristic
            x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)

            if open_costs[x, y] == np.inf:
                break

            parent = np.array([x, y])
            closed_flags[x, y] = 1

            neighbors = get_neighborhood(parent, occ_map.shape)
            for child in neighbors:
                cx, cy = child
                if closed_flags[cx, cy] == 1:
                    continue
                edge_cost = get_edge_cost(parent, child, occ_map)
                if edge_cost == np.inf:
                    continue
                new_cost = costs[parent[0], parent[1]] + edge_cost
                if new_cost < costs[cx, cy]:
                    costs[cx, cy] = new_cost
                    predecessors[cx, cy] = parent

        if not np.array_equal(parent, goal):
            return []

        path = []
        while predecessors[parent[0], parent[1]][0] >= 0:
            path.append(parent.copy())
            parent = predecessors[parent[0], parent[1]]
        path.append(start.copy())
        path.reverse()
        return path


class AEstrellaNode(Node):
    """Nodo ROS que escuchará mapa, pose estimada y objetivos para planificar rutas."""

    def __init__(self):
        super().__init__("a_estrella_node")
        self.planner = AEstrellaPlanner()

        self.map_ready = False
        self.occ_map = None
        self.map_resolution = None
        self.map_origin = (0.0, 0.0)

        self.current_pose: Optional[PoseStamped] = None
        self.goal_pose: Optional[PoseStamped] = None
        self.latest_scan: Optional[LaserScan] = None

        self.path_cells: List[np.ndarray] = []
        self.path_world: List[Tuple[float, float]] = []

        # Parámetros del controlador (Pure Pursuit)
        self.declare_parameter("linear_speed", 0.04)
        self.declare_parameter("lookahead_distance", 0.2)
        self.declare_parameter("goal_tolerance", 0.1)
        self.declare_parameter("heading_turn_threshold", 0.2)
        self.declare_parameter("yaw_tolerance", 0.1)
        self.declare_parameter("max_angular_speed", 0.12)
        self.declare_parameter("obstacle_distance", 0.3)
        self.declare_parameter("point_reached_distance", 0.05)
        self.declare_parameter("front_stop_distance", 0.25)
        self.declare_parameter("occupancy_threshold", 0.5)
        self.declare_parameter("skip_points_after_plan", 6)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.lookahead = float(self.get_parameter("lookahead_distance").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.heading_turn_threshold = float(self.get_parameter("heading_turn_threshold").value)
        self.yaw_tolerance = float(self.get_parameter("yaw_tolerance").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.obstacle_distance = float(self.get_parameter("obstacle_distance").value)
        self.point_reached_distance = float(self.get_parameter("point_reached_distance").value)
        self.front_stop_distance = float(self.get_parameter("front_stop_distance").value)
        self.occupancy_threshold = float(self.get_parameter("occupancy_threshold").value)
        self.skip_points_after_plan = int(self.get_parameter("skip_points_after_plan").value)

        self.state = "WAIT_DATA"
        self.goal_heading = 0.0
        self.filtered_heading_error = 0.0
        self.obstacle_detected = False
        self.front_blocked = False

        qos_transient = QoSProfile(depth=1)
        qos_transient.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.sub_map = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_transient)
        self.sub_pose = self.create_subscription(PoseStamped, "/estimated_pose", self.pose_callback, 10)
        self.sub_goal = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, "/planned_path", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.timer = self.create_timer(0.1, self.state_machine_loop)

        self.get_logger().info("Nodo A* listo (escuchando mapa, pose y goal).")

    def map_callback(self, msg: OccupancyGrid):
        """Convierte el OccupancyGrid en grilla de probabilidades para el planificador."""
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.float32).reshape((height, width)).T  # shape (width, height)

        prob = np.clip(data, 0, 100) / 100.0
        unknown_mask = data < 0
        prob[unknown_mask] = 1.0

        binary_map = np.zeros_like(prob)
        binary_map[prob >= self.occupancy_threshold] = 1.0

        self.occ_map = binary_map
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_ready = True
        self.get_logger().info(f"Mapa recibido en a_estrella (w={width}, h={height}).")

    def pose_callback(self, msg: PoseStamped):
        """Guarda la pose estimada publicada por el nodo de localización."""
        self.current_pose = msg

    def goal_callback(self, msg: PoseStamped):
        """Guarda el objetivo capturado con 2D Goal Pose."""
        self.goal_pose = msg
        q = msg.pose.orientation
        self.goal_heading = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.get_logger().info(
            f"Goal recibido: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """Convierte coordenadas del mundo en índices de grilla."""
        mx = int((x - self.map_origin[0]) / self.map_resolution)
        my = int((y - self.map_origin[1]) / self.map_resolution)
        return mx, my

    def try_plan(self):
        """Lanza A* si hay mapa, pose actual y objetivo disponibles."""
        if not self.map_ready or self.current_pose is None or self.goal_pose is None:
            self.get_logger().debug("No puedo planificar: faltan datos.")
            return

        start = self.world_to_map(
            self.current_pose.pose.position.x, self.current_pose.pose.position.y
        )
        goal = self.world_to_map(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)

        self.get_logger().info(f"Planificando desde {start} hasta {goal} (celdas).")
        path = self.planner.plan(self.occ_map, np.array(start), np.array(goal))
        if not path:
            self.get_logger().warn("No se encontró camino con A*. Esperando nuevo objetivo.")
            self.state = "WAIT_GOAL"
        else:
            self.get_logger().info(f"Camino A* obtenido con {len(path)} puntos.")
            self.path_cells = path
            self.path_world = [self.map_to_world(p[0], p[1]) for p in path]
            if len(self.path_world) > 1:
                skip = min(self.skip_points_after_plan, len(self.path_world) - 1)
                if skip > 0:
                    self.path_world = self.path_world[skip:]
            if len(self.path_world) >= 2:
                prev = self.path_world[-2]
                last = self.path_world[-1]
                auto_heading = math.atan2(last[1] - prev[1], last[0] - prev[0])
                if abs(self.goal_heading) < 1e-3:
                    self.goal_heading = auto_heading
            self.publish_path_msg()
            self.filtered_heading_error = 0.0
            self.state = "FOLLOWING"

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Convierte índices de grilla a coordenadas del mundo."""
        x = mx * self.map_resolution + self.map_origin[0]
        y = my * self.map_resolution + self.map_origin[1]
        return x, y

    def publish_path_msg(self):
        """Publica el path como nav_msgs/Path para visualizar en RViz."""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for px, py in self.path_world:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = px
            pose.pose.position.y = py
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def state_machine_loop(self):
        """Máquina de estados centralizada."""
        if not self.map_ready or self.current_pose is None:
            return

        if self.latest_scan is not None and self.occ_map is not None:
            self.evaluate_scan(self.latest_scan)

        if self.state == "WAIT_DATA":
            if self.goal_pose is not None:
                self.state = "PLANNING"
            else:
                return

        if self.state == "PLANNING":
            self.try_plan()
            return

        if self.state == "WAIT_GOAL":
            self.publish_stop()
            self.filtered_heading_error = 0.0
            return

        if self.state == "REPLAN":
            self.publish_stop()
            self.perform_reactive_maneuver()
            self.state = "PLANNING"
            return

        if self.state == "ALIGN_FINAL":
            self.align_heading()
            return

        if self.state == "FOLLOWING":
            self.run_pure_pursuit()
            return

    def run_pure_pursuit(self):
        if not self.path_world or self.current_pose is None:
            self.publish_stop()
            self.state = "WAIT_GOAL"
            return

        if self.front_blocked:
            self.publish_stop()
            self.state = "REPLAN"
            return

        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        target = self.get_lookahead_point(robot_x, robot_y)
        if target is None:
            self.get_logger().info("Objetivo alcanzado, esperando nuevo goal.")
            self.state = "WAIT_GOAL"
            self.path_world = []
            self.path_cells = []
            self.publish_stop()
            return

        lx = math.cos(-yaw) * (target[0] - robot_x) - math.sin(-yaw) * (target[1] - robot_y)
        ly = math.sin(-yaw) * (target[0] - robot_x) + math.cos(-yaw) * (target[1] - robot_y)

        distance = math.sqrt((target[0] - robot_x) ** 2 + (target[1] - robot_y) ** 2)
        if distance < self.goal_tolerance:
            self.get_logger().info("Posición alcanzada. Alineando orientación final.")
            self.state = "ALIGN_FINAL"
            self.publish_stop()
            return

        heading_error = math.atan2(ly, lx)
        self.filtered_heading_error = 0.7 * self.filtered_heading_error + 0.3 * heading_error
        if abs(self.filtered_heading_error) > self.heading_turn_threshold:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = max(
                min(1.0 * self.filtered_heading_error, self.max_angular_speed), -self.max_angular_speed
            )
            self.cmd_pub.publish(cmd)
            return

        curvature = 0.0 if self.lookahead == 0 else (2.0 * ly) / (self.lookahead ** 2)
        cmd = Twist()

        cmd.linear.x = self.linear_speed

        cmd.angular.z = curvature * self.linear_speed
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)

        self.cmd_pub.publish(cmd)

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def perform_reactive_maneuver(self):
        """Gira un instante para despegarse antes de replanificar."""
        cmd = Twist()
        cmd.linear.x = -0.02
        cmd.angular.z = self.max_angular_speed
        self.cmd_pub.publish(cmd)

    def evaluate_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        q = self.current_pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y

        front_obstacle = False
        for r, ang in zip(ranges, angles):
            if not np.isfinite(r) or r <= 0.0 or r > self.obstacle_distance:
                continue
            global_angle = yaw + ang
            obs_x = robot_x + r * math.cos(global_angle)
            obs_y = robot_y + r * math.sin(global_angle)
            mx, my = self.world_to_map(obs_x, obs_y)
            if 0 <= mx < self.occ_map.shape[0] and 0 <= my < self.occ_map.shape[1]:
                if self.occ_map[mx, my] < 0.3:
                    self.get_logger().warn("Obstáculo no mapeado detectado. Replanificando.")
                    self.state = "REPLAN"
                    self.path_world = []
                    self.path_cells = []
                    return
            if abs(ang) < 0.2 and r < self.front_stop_distance:
                front_obstacle = True

        self.front_blocked = front_obstacle

    def get_lookahead_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Busca el punto del path que está a lookahead metros."""
        while self.path_world and math.hypot(self.path_world[0][0] - x, self.path_world[0][1] - y) < self.point_reached_distance:
            self.path_world.pop(0)

        if not self.path_world:
            return None
        for px, py in self.path_world:
            dist = math.hypot(px - x, py - y)
            if dist >= self.lookahead:
                return (px, py)
        # Si no hay más puntos lejos, tomar el último
        return self.path_world[-1] if self.path_world else None

    def align_heading(self):
        """Alinea el yaw del robot con el objetivo final."""
        if self.goal_pose is None or self.current_pose is None:
            self.state = "WAIT_GOAL"
            return

        desired_yaw = self.goal_heading
        q_cur = self.current_pose.pose.orientation
        current_yaw = self.quaternion_to_yaw(q_cur.x, q_cur.y, q_cur.z, q_cur.w)
        yaw_error = desired_yaw - current_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        if abs(yaw_error) < self.yaw_tolerance:
            self.get_logger().info("Orientación final alcanzada. Esperando nuevo goal.")
            self.state = "WAIT_GOAL"
            self.path_world = []
            self.path_cells = []
            self.publish_stop()
            return

        cmd = Twist()
        cmd.angular.z = 0.6 * yaw_error
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        """Publica cmd_vel cero."""
        stop = Twist()
        self.cmd_pub.publish(stop)

    @staticmethod
    def quaternion_to_yaw(x, y, z, w) -> float:
        """Convierte quat a yaw."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = AEstrellaNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
