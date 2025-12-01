import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped, PoseArray, PoseStamped, Pose
from tf2_ros import TransformBroadcaster
import numpy as np
import math

# Constantes de Mapeo
MAP_SIZE_X = 20.0 # metros
MAP_SIZE_Y = 20.0 # metros
RESOLUTION = 0.05 # metros/pixel
MAP_CENTER_X = 10.0 # offset en metros
MAP_CENTER_Y = 10.0

# Probabilidades Log-Odds
L_OCC = np.log(0.8 / 0.2) # Log-odd de celda ocupada
L_FREE = np.log(0.3 / 0.7) # Log-odd de celda libre
L_THRESH = 2.0 # Umbral para decir que está "ocupado" definitivamente

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

class FastSlamNode(Node):
    def __init__(self):
        super().__init__('fast_slam_node')

        # No uso la clase Partcile para intentar optimizar el uso de partículas
        self.num_particles = 50
        self.particles = np.zeros((self.num_particles, 4)) # [x, y, theta, peso]
        self.particles[:, 3] = 1.0 / self.num_particles

        self.width = int(MAP_SIZE_X / RESOLUTION)
        self.height = int(MAP_SIZE_Y / RESOLUTION)
        self.map_log_odds = np.zeros((self.width, self.height))
        
        # Variables de estado
        self.last_odom = None
        self.robot_pose = [0.0, 0.0, 0.0] # Pose estimada final (x, y, theta)

        self.sub_odom = self.create_subscription(Odometry, '/calc_odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.pub_map = self.create_publisher(OccupancyGrid, '/map', 10)
        self.pub_particles = self.create_publisher(PoseArray, '/particle_cloud', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/estimated_pose', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Nodo FastSLAM iniciado")

    def odom_callback(self, msg):
        """
        Calcula el delta de movimiento y mueve todas las partículas.
        """
        # Obtener pose actual de odometría
        q = msg.pose.pose.orientation
        _, _, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y

        if self.last_odom is None:
            self.last_odom = (current_x, current_y, yaw)
            return

        # Calcular deltas relativos al robot (delta_trans, delta_rot1, delta_rot2)
        dx = current_x - self.last_odom[0]
        dy = current_y - self.last_odom[1]
        dyaw = yaw - self.last_odom[2]
        
        # Modelo simple de odometría (rotación + traslación)
        delta_trans = math.sqrt(dx**2 + dy**2)
        delta_rot = dyaw # Simplificación para movimientos pequeños

        # Actualizar partículas con Ruido Gaussiano
        noise_trans = 0.05 # Desviación estándar en metros
        noise_rot = 0.05   # Desviación estándar en radianes

        # Vectorización numpy para eficiencia
        # Nuevo X = X + (v + ruido) * cos(theta)
        noise_t = np.random.normal(0, noise_trans, self.num_particles)
        noise_r = np.random.normal(0, noise_rot, self.num_particles)

        self.particles[:, 0] += (delta_trans + noise_t) * np.cos(self.particles[:, 2])
        self.particles[:, 1] += (delta_trans + noise_t) * np.sin(self.particles[:, 2])
        self.particles[:, 2] += (delta_rot + noise_r)
        
        # Normalizar ángulos entre -pi y pi
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        self.last_odom = (current_x, current_y, yaw)

    def scan_callback(self, msg):
        """
        Corrección, resampling y mapeo
        """
        if self.last_odom is None: return # Espera a que haya una odometría

        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = msg.range_max # Filtrar infinitos y NaNs
        ranges = np.nan_to_num(ranges, nan=msg.range_max)
        
        # --- PASO 2: CORRECCIÓN (Sensor Model) ---
        # Calcular pesos de partículas comparando scan con mapa actual
        self.update_weights(ranges, msg.angle_min, msg.angle_increment)

        # Resampling
        self.resample_particles()

        # Estimar la mejor pose (promedio ponderado o la de mayor peso)
        best_idx = np.argmax(self.particles[:, 3])
        best_particle = self.particles[best_idx]
        self.robot_pose = best_particle[:3]

        # Actualizamos el mapa global usando SOLO la mejor partícula
        self.update_map(best_particle, ranges, msg.angle_min, msg.angle_increment)

        timestamp = msg.header.stamp
        self.publish_map(timestamp)
        self.publish_particles(timestamp) 
        self.publish_estimated_pose(timestamp)


    def update_weights(self, ranges, angle_min, angle_inc):
        """
        Calcula qué tan bien encaja el scan de cada partícula con el mapa construido.
        """
        # Si el mapa está vacío (al inicio), no podemos ponderar bien -> pesos uniformes
        if np.all(self.map_log_odds == 0):
            return 

        # Submuestreo del láser para rendimiento (usar 1 de cada 10 rayos)
        step = 10 
        valid_ranges = ranges[::step]
        angles = np.arange(len(ranges)) * angle_inc + angle_min
        valid_angles = angles[::step]

        for i in range(self.num_particles):
            x_p, y_p, theta_p = self.particles[i, :3]
            
            # Proyectar puntos del láser al mundo según esta partícula
            laser_x = x_p + valid_ranges * np.cos(theta_p + valid_angles)
            laser_y = y_p + valid_ranges * np.sin(theta_p + valid_angles)
            
            # Convertir a índices de grilla
            idx_x, idx_y = self.world_to_map(laser_x, laser_y)
            
            # Verificar límites
            valid_mask = (idx_x >= 0) & (idx_x < self.width) & (idx_y >= 0) & (idx_y < self.height)
            
            # Score: Sumar valores del mapa en donde caen los puntos.
            # Si caen en celdas ocupadas (>0), el score sube. Si caen en libres (<0), baja.
            # Nota: Este es un modelo de verosimilitud simple ("Endpoint Model")
            score = np.sum(self.map_log_odds[idx_x[valid_mask], idx_y[valid_mask]])
            
            # Convertir log-score a probabilidad (simplificado)
            # Sumamos un offset para evitar underflow y exponenciamos
            self.particles[i, 3] = np.exp(score * 0.1) # Factor 0.1 para suavizar

        # Normalizar pesos
        sum_weights = np.sum(self.particles[:, 3])
        if sum_weights > 0:
            self.particles[:, 3] /= sum_weights
        else:
            self.particles[:, 3] = 1.0 / self.num_particles # Reset si divergen

    def resample_particles(self):
        """
        Resampling estocástico universal (Low Variance Sampling es mejor, pero este es simple)
        """
        # Calcular N_eff para ver si es necesario resamplear
        weights = self.particles[:, 3]
        n_eff = 1.0 / np.sum(np.square(weights))

        if n_eff < self.num_particles / 2.0:
            indices = np.random.choice(
                self.num_particles, 
                self.num_particles, 
                p=weights
            )
            self.particles = self.particles[indices]
            self.particles[:, 3] = 1.0 / self.num_particles # Reset pesos post-resample

    def update_map(self, particle, ranges, angle_min, angle_inc):
        """
        Actualiza la grilla log-odds usando Bresenham (Raycasting)
        """
        x_rob, y_rob, theta_rob = particle[:3]
        
        # Coordenada robot en grilla
        r_cx, r_cy = self.world_to_map_single(x_rob, y_rob)
        
        # Submuestreo para mapeo (puede ser más denso que para localización)
        step = 5
        for i in range(0, len(ranges), step):
            r = ranges[i]
            # Ignorar lecturas máximas (no rebotó en nada) para ocupación, 
            # pero sí sirven para limpiar espacio (free space).
            
            angle = angle_min + i * angle_inc + theta_rob
            
            # Punto final del rayo (Obstáculo)
            end_x = x_rob + r * np.cos(angle)
            end_y = y_rob + r * np.sin(angle)
            
            e_cx, e_cy = self.world_to_map_single(end_x, end_y)
            
            # Raycasting usando algoritmo de Bresenham simplificado o librería
            # Aquí recorremos las celdas entre robot y obstáculo
            cells = self.bresenham(r_cx, r_cy, e_cx, e_cy)
            
            for (cx, cy) in cells[:-1]: # Todas menos la última son LIBRES
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    self.map_log_odds[cx, cy] += L_FREE

            # La última celda es OCUPADA (si r < max_range)
            if r < 3.5: # Asumiendo max range del turtlebot
                if 0 <= e_cx < self.width and 0 <= e_cy < self.height:
                    self.map_log_odds[e_cx, e_cy] += L_OCC

        np.clip(self.map_log_odds, -100, 100, out=self.map_log_odds)

    def world_to_map(self, wx, wy):
        mx = ((wx + MAP_CENTER_X) / RESOLUTION).astype(int)
        my = ((wy + MAP_CENTER_Y) / RESOLUTION).astype(int)
        return mx, my
    
    def world_to_map_single(self, wx, wy):
        mx = int((wx + MAP_CENTER_X) / RESOLUTION)
        my = int((wy + MAP_CENTER_Y) / RESOLUTION)
        return mx, my

    def bresenham(self, x0, y0, x1, y1):
        """Genera puntos de grilla entre dos coordenadas"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))
        return points

    def publish_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.info.resolution = RESOLUTION
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = -MAP_CENTER_X
        msg.info.origin.position.y = -MAP_CENTER_Y
        msg.info.origin.orientation.w = 1.0
        
        # Convertir Log-Odds a Probabilidad [0, 100]
        # p = 1 - 1 / (1 + exp(l))
        probs = 1.0 - 1.0 / (1.0 + np.exp(self.map_log_odds))
        grid_data = (probs * 100).astype(np.int8)
        
        # Marcar desconocidos (cerca de 0 log odds -> p=0.5 -> valor 50)
        # En ROS, -1 es desconocido. Podemos hacer un umbral.
        unknown_mask = (np.abs(self.map_log_odds) < 0.1)
        grid_data[unknown_mask] = -1
        
        # ROS usa row-major order, flattened
        # Transponer porque numpy es (x,y) y OccupancyGrid se llena por filas
        msg.data = grid_data.T.flatten().tolist()
        
        self.pub_map.publish(msg)

    def publish_particles(self, stamp):
        """Publica todas las partículas como una nube de flechas en Rviz"""
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "map" # Las partículas existen en el mapa global

        for p in self.particles:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            # Convertir theta (yaw) a Quaternion
            q = euler_to_quaternion(0, 0, p[2])
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            msg.poses.append(pose)
        
        self.pub_particles.publish(msg)

    def publish_estimated_pose(self, stamp):
        """Publica la mejor estimación como una flecha grande"""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        
        msg.pose.position.x = self.robot_pose[0]
        msg.pose.position.y = self.robot_pose[1]
        q = euler_to_quaternion(0, 0, self.robot_pose[2])
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        
        self.pub_pose.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FastSlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()