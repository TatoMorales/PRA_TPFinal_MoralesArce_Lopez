import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# Turtlebot3
T_lidar_turtle3 = np.eye(4)

# Turtlebot4
T_lidar_turtle4 = np.array([
    [0.0, -1.0, 0.0, -0.04],
    [1.0,  0.0, 0.0,  0.00],
    [0.0,  0.0, 1.0,  0.193],
    [0.0,  0.0, 0.0,  1.0]
])

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

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def transform_lidar_to_robot(points_lidar, T_lidar_robot):
    """Transforma coordenadas del LIDAR a la terna del robot."""
    ones = np.ones((1, points_lidar.shape[1]))
    points_hom = np.vstack((points_lidar, ones))  # (4, N)
    points_robot = T_lidar_robot @ points_hom
    return points_robot[:3, :]  # devuelve solo x, y, z


class ObstacleNavigationNode(Node):
    def __init__(self):
        super().__init__('obstacle_navigation')
        self.declare_parameter('robot_model', 'turtlebot3')
        model = self.get_parameter('robot_model').value.lower()

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scans_sub = self.create_subscription(LaserScan, '/scan', self.scans_callback, 10)
        self.calc_odom_sub = self.create_subscription(Odometry, '/calc_odom', self.odom_callback, 10)

        self.turn_angle = np.deg2rad(110)
        self.threshold_distance = 0.5
        self.state = "FORWARD"
        self.current_theta = 0.0
        self.target_theta = None
        if model == 'turtlebot4':
            self.T_lidar_robot = T_lidar_turtle4
        else:
            self.T_lidar_robot = T_lidar_turtle3
        self.scan_ranges = []
        self.timer = self.create_timer(0.1, self.states_machine_loop)

        self.get_logger().info(f"Nodo de navegación listo, versión para {model}")

    def states_machine_loop(self):
        twist = Twist()

        if self.state == "FORWARD":
            twist.linear.x = 0.5
            twist.angular.z = 0.0

        elif self.state == "TURNING":
            angle_diff = normalize_angle(self.target_theta - self.current_theta)
            if abs(angle_diff) > 0.05: # Si está a más de 0.05 rads de los 110 grados objetivo, sigue rotando.
                twist.linear.x = 0.0
                twist.angular.z = 1.0
            else:
                self.get_logger().info("Giro completado — retomando avance.")
                self.state = "FORWARD"
                self.target_theta = None

        self.cmd_vel_pub.publish(twist)


    # def scans_callback(self, msg: LaserScan):
    #     """Se fija a cuanto está de un obstáculo"""
    #     try: 
    #         angles = np.linspace(msg.angle_min, msg.angle_max, np.shape(msg.ranges)[0], endpoint='true')
    #         angles = [normalize_angle(a) for a in angles]
    #         sector_min = -np.pi / 5
    #         sector_max = np.pi / 5
    #         self.get_logger().info(f"Angles size: {len(angles)}")
    #         interest_angles = [True if sector_min <= angle <= sector_max else False for angle in angles]
    #         self.get_logger().info(f"Interest angles size: {len(interest_angles)}")
    #         interest_ranges = [] 
    #         for index, valid in enumerate(interest_angles): 
    #             if valid: interest_ranges.append(msg.ranges[index]) 
    #         final_ranges = [range for range in interest_ranges if np.isfinite(range) & (range >= msg.range_min) & (range <= msg.range_max)]
    #         self.get_logger().info(f"Final ranges size: {len(final_ranges)}")
    #         final_ranges_local = transform_lidar_to_robot(final_ranges, self.T_lidar_robot)
    #         if self.state == "FORWARD":
    #             for range in final_ranges_local:
    #                 if range <= self.threshold_distance:
    #                     self.get_logger().info(f"Obstáculo a menos de {self.threshold_distance}mts detectado, iniciado giro")
    #                     self.state = "TURNING"
    #                     self.target_theta = normalize_angle(self.current_theta + self.turn_angle)
    #     except Exception as e:
    #          self.get_logger().error(f"Error: {e}")
    
    def scans_callback(self, msg: LaserScan):
        """Procesa los rangos del LIDAR, transforma a la terna del robot y detecta obstáculos frontales."""
        try:
            angles = [msg.angle_min + i * msg.angle_increment for i in range(len(msg.ranges))]
            angles = [normalize_angle(a) for a in angles]

            # Definimos el sector de interés: ±36° (aprox. π/5 rad)
            sector_min = -np.pi / 5
            sector_max = np.pi / 5

            # Seleccionamos los rayos dentro de ese sector
            interest_points = []
            for i, angle in enumerate(angles):
                r = msg.ranges[i]
                if sector_min <= angle <= sector_max and np.isfinite(r) and msg.range_min <= r <= msg.range_max:
                    # Puntos en coordenadas del LIDAR (plano 2D, z=0)
                    x_l = r * np.cos(angle)
                    y_l = r * np.sin(angle)
                    interest_points.append([x_l, y_l, 0.0])

            if not interest_points:
                return  # no hay datos útiles

            # --- Transformar puntos del LIDAR a la terna del robot ---
            points_lidar = np.array(interest_points).T  # (3, N)
            points_robot = transform_lidar_to_robot(points_lidar, self.T_lidar_robot)  # (3, N)

            # --- Convertir de nuevo a coordenadas polares en el marco del robot ---
            ranges_robot = []
            for i in range(points_robot.shape[1]):
                x_r, y_r = points_robot[0, i], points_robot[1, i]
                r_robot = np.sqrt(x_r**2 + y_r**2)
                ranges_robot.append(r_robot)

            # --- Detectar obstáculos ---
            if self.state == "FORWARD":
                if ranges_robot and min(ranges_robot) <= self.threshold_distance:
                    self.get_logger().info(
                        f"Obstáculo detectado a {min(ranges_robot):.2f} m — iniciando giro"
                    )
                    self.state = "TURNING"
                    self.target_theta = normalize_angle(self.current_theta + self.turn_angle)

        except Exception as e:
            self.get_logger().error(f"Error en scans_callback: {e}")

    def odom_callback(self, msg: Odometry):
        try:
            q = msg.pose.pose.orientation # Según la documentación de ros2 el campo orientation del mensaje Odometry es un cuaternion
            _, _, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
            self.current_theta = yaw

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
    
def main(args=None):
    rclpy.init(args=args)
    node = ObstacleNavigationNode()
    rclpy.spin(node)
    node
