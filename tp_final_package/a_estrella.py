import math
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan


# Reutilizado de planning_framework.py: funciones auxiliares del algoritmo A*
def get_neighborhood(cell: Tuple[int, int], occ_map_shape: Tuple[int, int]) -> List[List[int]]:
    """Vecinos usando coordenadas (x, y)."""
    neighbors: List[List[int]] = []
    x, y = cell
    nx, ny = occ_map_shape  # width, height

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx_ = x + dx
            ny_ = y + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny:
                neighbors.append([nx_, ny_])

    return neighbors



def get_heuristic(cell: Tuple[int, int], goal: Tuple[int, int]) -> float:
    dx = float(cell[0] - goal[0])
    dy = float(cell[1] - goal[1])
    return math.sqrt(dx * dx + dy * dy)


class AEstrellaPlanner:
    """Versión sin ROS del framework de planificación para reutilizar dentro del nodo."""

    def __init__(self, inflation_radius_cells: int = 1):
        self.inflation_radius_cells = inflation_radius_cells

    def is_cell_safe(self, cell: Tuple[int, int], occ_map: np.ndarray) -> bool:
        cx, cy = cell
        r = self.inflation_radius_cells
        nx, ny = occ_map.shape
        x0 = max(cx - r, 0)
        x1 = min(cx + r + 1, nx)
        y0 = max(cy - r, 0)
        y1 = min(cy + r + 1, ny)
        return np.all(occ_map[x0:x1, y0:y1] == 0.0)

    def edge_cost(self, parent: Tuple[int, int], child: Tuple[int, int], occ_map: np.ndarray) -> float:
        if not self.is_cell_safe(child, occ_map):
            return np.inf
        px, py = parent
        cx, cy = child
        dx = px - cx
        dy = py - cy
        return 1.0 if (dx == 0 or dy == 0) else math.sqrt(2.0)

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
                edge_cost = self.edge_cost(parent, child, occ_map)
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
        self.declare_parameter("linear_speed", 0.05)
        self.declare_parameter("lookahead_distance", 0.2)
        self.declare_parameter("goal_tolerance", 0.1)
        self.declare_parameter("heading_turn_threshold", 0.4)
        self.declare_parameter("yaw_tolerance", 0.1)
        self.declare_parameter("max_angular_speed", 0.12)
        self.declare_parameter("obstacle_distance", 0.20)
        self.declare_parameter("point_reached_distance", 0.05)
        self.declare_parameter("front_stop_distance", 0.18)
        self.declare_parameter("occupancy_threshold", 0.5)
        self.declare_parameter("inflation_radius", 0.3)
        self.declare_parameter("skip_points_after_plan", 2)
        self.declare_parameter("block_detection_cycles", 3)
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
        self.inflation_radius = float(self.get_parameter("inflation_radius").value)
        self.skip_points_after_plan = int(self.get_parameter("skip_points_after_plan").value)
        self.block_detection_cycles = int(self.get_parameter("block_detection_cycles").value)

        self.state = "WAIT_DATA"
        self.goal_heading = 0.0
        self.filtered_heading_error = 0.0
        self.obstacle_detected = False
        self.front_blocked = False
        self.block_counter = 0
        self.front_block_cycles = 0
        self.replan_due_to_front = False
        self.reactive_active = False
        self.reactive_phase = 0
        self.reactive_phase_end = None
        self.reactive_sequence = [
            ("backward", 0.4),
            ("rotate", 0.4),
        ]

        self.planner = AEstrellaPlanner()

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
        # tratamos unknown como ocupado para que no planifique por ahí
        prob[unknown_mask] = 1.0

        binary_map = np.zeros_like(prob)
        binary_map[prob >= self.occupancy_threshold] = 1.0  # 1 = ocupado/unknown, 0 = libre

        self.occ_map = binary_map  # shape (width, height)
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        inflation_cells = int(max(0, math.ceil(self.inflation_radius / self.map_resolution)))
        self.planner.inflation_radius_cells = max(0, inflation_cells)
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

        # Reset de cosas del path anterior
        self.path_world = []
        self.path_cells = []
        self.reactive_active = False
        self.block_counter = 0

        # Si ya tengo mapa y pose, paso directamente a planificar
        if self.map_ready and self.current_pose is not None:
            self.state = "PLANNING"
        else:
            self.state = "WAIT_DATA"


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

        start = tuple(self.world_to_map(
            self.current_pose.pose.position.x, self.current_pose.pose.position.y
        ))
        goal = tuple(self.world_to_map(
            self.goal_pose.pose.position.x, self.goal_pose.pose.position.y
        ))

        if not (self.is_index_in_map(start) and self.is_index_in_map(goal)):
            self.get_logger().warn("Start o goal fuera del mapa, reubicar y reintentar.")
            self.state = "WAIT_DATA"
            return

        self.get_logger().info(f"Planificando desde {start} hasta {goal} (celdas [x,y]).")
        backup_radius = self.planner.inflation_radius_cells
        path = None
        for radius in range(backup_radius, -1, -1):
            self.planner.inflation_radius_cells = radius
            if not (self.planner.is_cell_safe(start, self.occ_map) and self.planner.is_cell_safe(goal, self.occ_map)):
                continue
            candidate = self.planner.plan(self.occ_map, np.array(start), np.array(goal))
            if candidate:
                path = candidate
                if radius < backup_radius:
                    self.get_logger().warn(
                        f"Inflación reducida a {radius} celdas para encontrar camino."
                    )
                break
        self.planner.inflation_radius_cells = backup_radius

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

    def map_to_world(self, x: int, y: int) -> Tuple[float, float]:
        """Convierte índices de grilla a coordenadas del mundo."""
        wx = x * self.map_resolution + self.map_origin[0]
        wy = y * self.map_resolution + self.map_origin[1]
        return wx, wy

    def is_index_in_map(self, cell: Tuple[int, int]) -> bool:
        x, y = cell
        nx, ny = self.occ_map.shape
        return 0 <= x < nx and 0 <= y < ny

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
            if self.replan_due_to_front:
                if not self.reactive_active:
                    self.start_reactive_maneuver()
                self.process_reactive_maneuver()
                if not self.reactive_active:
                    self.replan_due_to_front = False
                    self.state = "PLANNING"
            else:
                self.publish_stop()
                self.state = "PLANNING"
            return

        if self.state == "ALIGN_FINAL":
            self.align_heading()
            return
        
        if self.state == "FOLLOWING":
            if self.latest_scan is not None and self.occ_map is not None:
                self.evaluate_scan(self.latest_scan)
                # Si evaluate_scan decidió replanificar, no seguimos el path actual
                if self.state == "REPLAN":
                    return
            else:
                self.block_counter = 0

            self.run_pure_pursuit()
            return
        else:
            self.block_counter = 0

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
            self.publish_path_msg()
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
                min(1.0 * self.filtered_heading_error, self.max_angular_speed),
                -self.max_angular_speed
            )
            self.cmd_pub.publish(cmd)
            return

        curvature = 0.0 if self.lookahead == 0 else (2.0 * ly) / (self.lookahead ** 2)
        cmd = Twist()

        # Limito la curvatura máxima para que no dé órdenes imposibles
        max_curv = 1.5
        curvature = max(min(curvature, max_curv), -max_curv)

        # Si el error de rumbo es MUY grande, mejor girar casi en el lugar
        abs_err = abs(self.filtered_heading_error)
        if abs_err > 0.8:  # ~45 grados
            cmd.linear.x = 0.0
            cmd.angular.z = max(
                min(1.0 * self.filtered_heading_error, self.max_angular_speed),
                -self.max_angular_speed,
            )
            self.cmd_pub.publish(cmd)
            return

        # Si el error no es tan grande, avanzo, pero más lento cuanto más curva sea
        base_v = self.linear_speed
        # factor entre 0.3 y 1 según cuánta curvatura necesito
        speed_factor = max(0.3, 1.0 - abs(curvature) / max_curv)
        cmd.linear.x = base_v * speed_factor

        cmd.angular.z = curvature * cmd.linear.x
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)

        self.cmd_pub.publish(cmd)


    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def start_reactive_maneuver(self):
        """Inicializa la maniobra reactiva para despegarse del obstáculo."""
        self.reactive_active = True
        self.reactive_phase = 0
        duration = self.reactive_sequence[0][1]
        self.reactive_phase_end = self.get_clock().now() + Duration(seconds=duration)

    def process_reactive_maneuver(self):
        """Ejecuta la maniobra reactiva en pasos temporizados."""
        if not self.reactive_active:
            return

        phase_name, _ = self.reactive_sequence[self.reactive_phase]
        cmd = Twist()
        if phase_name == "backward":
            cmd.linear.x = -self.linear_speed
        elif phase_name == "rotate":
            cmd.angular.z = self.max_angular_speed * 0.8
        self.cmd_pub.publish(cmd)

        now = self.get_clock().now()
        if now >= self.reactive_phase_end:
            self.reactive_phase += 1
            if self.reactive_phase >= len(self.reactive_sequence):
                self.reactive_active = False
                self.publish_stop()
                return
            duration = self.reactive_sequence[self.reactive_phase][1]
            self.reactive_phase_end = now + Duration(seconds=duration)

    def evaluate_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        q = self.current_pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y

        front_obstacle = False
        new_obstacle = False
        map_w, map_h = self.occ_map.shape
        for r, ang in zip(ranges, angles):
            if not np.isfinite(r) or r <= 0.0:
                continue
            global_angle = yaw + ang
            obs_x = robot_x + r * math.cos(global_angle)
            obs_y = robot_y + r * math.sin(global_angle)
            mx, my = self.world_to_map(obs_x, obs_y)
            inside_map = 0 <= mx < map_w and 0 <= my < map_h
            cell_is_free = False
            if inside_map:
                cell_is_free = self.occ_map[mx, my] == 0.0
                if cell_is_free and r < self.obstacle_distance:
                    new_obstacle = True
            # solo consideramos bloqueo frontal si el rayo cae en una celda libre (obstáculo no mapeado)
            # o directamente fuera del mapa (desconocido)
            if abs(ang) < 0.2 and r < self.front_stop_distance and (not inside_map or cell_is_free):
                front_obstacle = True

        self.front_blocked = front_obstacle

        if front_obstacle:
            self.front_block_cycles += 1
        else:
            self.front_block_cycles = 0

        if new_obstacle:
            self.block_counter += 1
        else:
            self.block_counter = 0

        trigger_front = self.front_block_cycles >= 2
        trigger_new = self.block_counter >= self.block_detection_cycles

        if trigger_front or trigger_new:
            if trigger_front:
                self.front_block_cycles = 0
            if trigger_new:
                self.block_counter = 0
            reason = "Obstáculo frontal" if trigger_front else "Obstáculo nuevo"
            self.get_logger().warn(f"{reason} detectado. Replanificando.")
            self.replan_due_to_front = trigger_front
            self.state = "REPLAN"
            self.path_world = []
            self.path_cells = []
            self.publish_path_msg()
            return

    def get_lookahead_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Selecciona dinámicamente el waypoint objetivo para Pure Pursuit."""
        while self.path_world and math.hypot(self.path_world[0][0] - x,
                                            self.path_world[0][1] - y) < self.point_reached_distance:
            self.path_world.pop(0)

        if not self.path_world:
            return None

        goal_x, goal_y = self.path_world[-1]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)
        # Cerca del objetivo reducimos el lookahead para no cortar esquinas.
        effective_lookahead = min(
            self.lookahead,
            max(self.point_reached_distance * 1.5, 0.6 * dist_to_goal),
        )

        selected_idx = None
        nearest_idx = 0
        nearest_dist = float("inf")
        for idx, (px, py) in enumerate(self.path_world):
            dist = math.hypot(px - x, py - y)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
            if selected_idx is None and dist >= effective_lookahead:
                selected_idx = idx

        if selected_idx is None:
            selected_idx = len(self.path_world) - 1

        # Si estamos muy desviados, mejor volver al punto más cercano antes de seguir avanzando.
        if nearest_dist > effective_lookahead * 1.2:
            selected_idx = nearest_idx

        if selected_idx > 0:
            self.path_world = self.path_world[selected_idx:]

        return self.path_world[0] if self.path_world else None

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
