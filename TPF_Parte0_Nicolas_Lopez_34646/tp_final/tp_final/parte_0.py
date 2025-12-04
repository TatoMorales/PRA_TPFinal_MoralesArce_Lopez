from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import rclpy
import numpy as np
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.USE_TB4 =  False

        if self.USE_TB4:
            self.LIDAR_YAW_OFFSET = np.pi / 2
            self.ROT_ANGLE = (110.0 - 20.5) * np.pi / 180.0 
            self.FILTER_INTENSITY = True
        else:
            self.LIDAR_YAW_OFFSET = 0.0
            self.ROT_ANGLE = 110.0 * np.pi / 180.0
            self.FILTER_INTENSITY = False

        self.obstaculo = False
        self.yaw = 0.0
        self.yaw_ref = None
        self.estado = 'AVANZAR' 

        self.subscription_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.005, self.me_callback)

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def quat_to_yaw(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    def me_callback(self):
        msg = Twist()
        if self.estado == 'GIRAR':
            msg.linear.x  = 0.0
            msg.angular.z = 1.0
            if not self.obstaculo:
                self.estado = 'AVANZAR'

        elif self.estado == 'AVANZAR':
            msg.linear.x  = 0.3
            msg.angular.z = 0.0
            if self.obstaculo:
                self.estado = 'GIRAR'
        self.publisher_cmd.publish(msg)

    def odom_callback(self, data: Odometry):
        self.yaw = self.quat_to_yaw(data.pose.pose.orientation)
        if self.obstaculo and (self.yaw_ref is not None):
            err = self.normalize_angle((self.yaw_ref + self.ROT_ANGLE) - self.yaw)
            if abs(err) < 0.03:
                self.yaw_ref = None
                self.obstaculo = False

    def scan_callback(self, data: LaserScan):
        ranges = np.asarray(data.ranges, dtype=np.float32)
        n = len(ranges)
        angles = data.angle_min + np.arange(n, dtype=np.float32) * data.angle_increment

        angles_robot = angles + self.LIDAR_YAW_OFFSET

        mask_front = (angles_robot >= -np.pi/4) & (angles_robot <= np.pi/4)
        valid = np.isfinite(ranges) & (ranges >= data.range_min) & (ranges <= data.range_max)

        if self.FILTER_INTENSITY and len(data.intensities) == n:
            intensities = np.asarray(data.intensities, dtype=np.float32)
            valid = valid & (intensities != 0.0)

        front = ranges[mask_front & valid]

        if (not self.obstaculo) and (front.size > 0) and (np.min(front) <= 0.5):
            self.obstaculo = True
            self.yaw_ref = self.yaw
            if np.min(front) > 0.5:
                self.obstaculo = False

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
 