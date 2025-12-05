import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseArray, PoseStamped, Pose, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

# Reutilizado de fast_slam.py: utilidades para conversiones
def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


class FastSlamLocalizationNode(Node):
    def __init__(self):
        super().__init__("fast_slam_localization")

        # Reutilizado de fast_slam.py: cantidad de partículas y estructura [x, y, theta, peso]
        self.num_particles = 80
        self.particles = np.zeros((self.num_particles, 4))
        self.particles[:, 3] = 1.0 / self.num_particles

        # Nuevo: la localización usa un mapa estático recibido en /map
        self.map_ready = False
        self.map_resolution = None
        self.map_origin = (0.0, 0.0)
        self.map_log_odds = None
        self.map_width = 0
        self.map_height = 0

        # Variables reutilizadas para manejar odometría y estimación final
        self.last_odom = None
        self.current_odom_raw = None
        self.robot_pose = [0.0, 0.0, 0.0]

        # Subscriciones compartidas con fast_slam (odom y láser)
        self.sub_odom = self.create_subscription(Odometry, "/calc_odom", self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # Nuevo: escuchar el mapa y la pose inicial desde RViz
        qos_transient = QoSProfile(depth=1)
        qos_transient.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.sub_map = self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_transient)
        self.sub_initialpose = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initialpose_callback, 10)

        # Publicadores heredados de fast_slam (pose estimada + nube de partículas)
        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_pose = self.create_publisher(PoseStamped, "/estimated_pose", 10)
        self.pub_particles = self.create_publisher(PoseArray, "/particle_cloud", 10)

        # Nuevo: este nodo no publica OccupancyGrid, solo el TF map->odom
        self.tf_broadcaster = TransformBroadcaster(self)

        self.initialized = False  # Nuevo: indica si recibimos un initialpose válido
        #self.get_logger().info("Nodo de localización FastSLAM listo. Esperando mapa e initialpose.")

    def map_callback(self, msg: OccupancyGrid):
        """Nuevo: carga el mapa estático publicado por map_server."""
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        data = np.array(msg.data, dtype=np.float32).reshape((self.map_height, self.map_width))
        log_odds = np.zeros_like(data)
        known_mask = data != -1
        if np.any(known_mask):
            # Reutilizado: usamos log-odds como en fast_slam para comparar lecturas
            prob = data[known_mask] / 100.0
            prob = np.clip(prob, 0.01, 0.99)
            log_odds[known_mask] = np.log(prob / (1.0 - prob))

        self.map_log_odds = log_odds
        self.map_ready = True
        #self.get_logger().info("Mapa estático recibido, ancho=%d alto=%d", self.map_width, self.map_height)

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        """Nuevo: inicializa la nube de partículas alrededor de la pose seleccionada en RViz."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)

        noise_pos = np.random.normal(0.0, 0.05, size=(self.num_particles, 2))
        noise_yaw = np.random.normal(0.0, 0.02, size=(self.num_particles,))

        self.particles[:, 0] = x + noise_pos[:, 0]
        self.particles[:, 1] = y + noise_pos[:, 1]
        self.particles[:, 2] = yaw + noise_yaw
        self.particles[:, 3] = 1.0 / self.num_particles

        self.robot_pose = [x, y, yaw]
        self.initialized = True
        #self.get_logger().info("Initial pose recibida: (%.2f, %.2f, %.2f rad)", x, y, yaw)

    def odom_callback(self, msg: Odometry):
        """Reutilizado: propagación de partículas mediante el modelo de movimiento."""
        q = msg.pose.pose.orientation
        _, _, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y

        if self.last_odom is None:
            self.last_odom = (current_x, current_y, yaw)
            self.current_odom_raw = (current_x, current_y, yaw)
            return

        dx = current_x - self.last_odom[0]
        dy = current_y - self.last_odom[1]
        dyaw = yaw - self.last_odom[2]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi

        delta_trans = math.sqrt(dx ** 2 + dy ** 2)
        noise_trans = 0.02
        noise_rot = 0.02

        noise_t = np.random.normal(0, noise_trans, self.num_particles)
        noise_r = np.random.normal(0, noise_rot, self.num_particles)

        self.particles[:, 0] += (delta_trans + noise_t) * np.cos(self.particles[:, 2])
        self.particles[:, 1] += (delta_trans + noise_t) * np.sin(self.particles[:, 2])
        self.particles[:, 2] += dyaw + noise_r
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        self.last_odom = (current_x, current_y, yaw)
        self.current_odom_raw = (current_x, current_y, yaw)

    def scan_callback(self, msg: LaserScan):
        """Reutilizado + nuevo: actualización de pesos usando el mapa fijo."""
        if not self.initialized or not self.map_ready or self.last_odom is None:
            return

        ranges = np.array(msg.ranges)
        ranges[ranges == float("inf")] = msg.range_max
        ranges = np.nan_to_num(ranges, nan=msg.range_max)

        self.update_weights(ranges, msg.angle_min, msg.angle_increment)
        self.resample_particles()

        best_idx = np.argmax(self.particles[:, 3])
        best_particle = self.particles[best_idx]
        self.robot_pose = best_particle[:3]

        stamp = msg.header.stamp
        self.publish_particles(stamp)
        self.publish_estimated_pose(stamp)
        self.publish_tf(stamp)

    def update_weights(self, ranges, angle_min, angle_inc):
        """Reutilizado de fast_slam: endpoint model comparado contra el mapa."""
        if self.map_log_odds is None:
            return

        step = 10
        valid_ranges = ranges[::step]
        angles = np.arange(len(ranges)) * angle_inc + angle_min
        valid_angles = angles[::step]

        for i in range(self.num_particles):
            x_p, y_p, theta_p = self.particles[i, :3]
            laser_x = x_p + valid_ranges * np.cos(theta_p + valid_angles)
            laser_y = y_p + valid_ranges * np.sin(theta_p + valid_angles)

            idx_x, idx_y = self.world_to_map(laser_x, laser_y)
            valid_mask = (
                (idx_x >= 0)
                & (idx_x < self.map_width)
                & (idx_y >= 0)
                & (idx_y < self.map_height)
            )

            score = np.sum(self.map_log_odds[idx_y[valid_mask], idx_x[valid_mask]])
            self.particles[i, 3] = np.exp(score * 0.1)

        sum_weights = np.sum(self.particles[:, 3])
        if sum_weights > 0:
            self.particles[:, 3] /= sum_weights
        else:
            self.particles[:, 3] = 1.0 / self.num_particles

    def resample_particles(self):
        """Reutilizado: muestreo según pesos."""
        weights = self.particles[:, 3]
        n_eff = 1.0 / np.sum(np.square(weights))
        if n_eff < self.num_particles / 2.0:
            indices = np.random.choice(self.num_particles, self.num_particles, p=weights)
            self.particles = self.particles[indices]
            self.particles[:, 3] = 1.0 / self.num_particles

    def publish_particles(self, stamp):
        """Reutilizado: publicar PoseArray para RViz."""
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        for p in self.particles:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            q = euler_to_quaternion(p[2], 0, 0)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            msg.poses.append(pose)
        self.pub_particles.publish(msg)

    def publish_estimated_pose(self, stamp):
        """Reutilizado: publicar mejor estimación."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = self.robot_pose[0]
        msg.pose.position.y = self.robot_pose[1]
        q = euler_to_quaternion(self.robot_pose[2], 0, 0)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pub_pose.publish(msg)

    def publish_tf(self, stamp):
        """Reutilizado: calcula TF map->odom mezclando estimado y odometría cruda."""
        if self.current_odom_raw is None:
            return

        x_map, y_map, theta_map = self.robot_pose
        x_odom, y_odom, theta_odom = self.current_odom_raw

        c_m = np.cos(theta_map)
        s_m = np.sin(theta_map)
        T_map_base = np.array(
            [
                [c_m, -s_m, 0, x_map],
                [s_m, c_m, 0, y_map],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        c_o = np.cos(theta_odom)
        s_o = np.sin(theta_odom)
        T_odom_base = np.array(
            [
                [c_o, -s_o, 0, x_odom],
                [s_o, c_o, 0, y_odom],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        T_base_odom = np.linalg.inv(T_odom_base)
        T_map_odom = np.dot(T_map_base, T_base_odom)

        tx = T_map_odom[0, 3]
        ty = T_map_odom[1, 3]
        yaw_correction = np.arctan2(T_map_odom[1, 0], T_map_odom[0, 0])
        q = euler_to_quaternion(yaw_correction, 0, 0)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def world_to_map(self, wx, wy):
        """Nuevo: usa origen y resolución del OccupancyGrid."""
        mx = ((wx - self.map_origin[0]) / self.map_resolution).astype(int)
        my = ((wy - self.map_origin[1]) / self.map_resolution).astype(int)
        return mx, my


def main(args=None):
    rclpy.init(args=args)
    node = FastSlamLocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
