import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.node import Node
from uwb_interfaces.msg import UwbRange
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance, PointStamped

import numpy as np
from scipy.optimize import least_squares
import time

MAX_NUM_ANCHORS = 4

def multilateration(distances):
    # Anchor positions
    anchors = np.array([
        [0.0, 0.0, 0.94],
        [1.683, 0.0, 1.75],
        [1.683, 1.205, 2.06],
        [0.0, 1.205, 1.33]
    ])

    # Function to minimize
    def residuals(variables):
        x, y, z = variables
        res = [
            np.sqrt((x - anchors[i][0])**2 + (y - anchors[i][1])**2 + (z - anchors[i][2])**2) - distances[i]
            for i in range(MAX_NUM_ANCHORS)
        ]
        return res

    # Initial guess
    initial_guess = [1.0, 1.0, 1.0]

    # Solve the system of equations
    solution = least_squares(residuals, initial_guess)
    position = solution.x

    return position

class KalmanFilter3DPositionOnly:
    def __init__(self, Q, R, P, x):
        self.Q = Q  # Process noise covariance
        self.R = R  # Observation noise covariance
        self.P = P  # Estimate error covariance
        self.x = x  # Initial state estimate

    def predict(self, dt):
        # Define the state transition matrix with dynamic dt
        F = np.array([[1, 0, 0],  # State transition matrix
                      [0, 1, 0],
                      [0, 0, 1]])
        
        # Predict the state and estimate error covariance
        self.x = np.dot(F, self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q
        return self.x

    def update(self, z):
        # Define the observation matrix
        H = np.array([[1, 0, 0],  # Observation matrix
                      [0, 1, 0],
                      [0, 0, 1]])
        
        # Compute the Kalman Gain
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(np.dot(np.dot(H, self.P), H.T) + self.R))
        
        # Update the state estimate and error covariance
        self.x = self.x + np.dot(K, (z - np.dot(H, self.x)))
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        return self.x

class UwbTagLocalizer(Node):
    def __init__(self):
        super().__init__('uwb_tag_localizer')
        self.get_logger().info('UwbTagLocalizer node started')
        
        #  set qos best effort
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.uwb_range_sub = self.create_subscription(
            UwbRange,
            'tag2___uros_uwb_tag_range',
            self.uwb_range_callback,
            qos_profile
        )

        self.uwb_point_raw_pub = self.create_publisher(
            PointStamped,
            'uwb_tag_point/raw',
            10
        )

        self.uwb_point_ekf_pub = self.create_publisher(
            PointStamped,
            'uwb_tag_point/ekf',
            10
        )

        self.uwb_odom_pub = self.create_publisher(
            Odometry,
            'uwb_tag_odom',
            10
        )

        # Initialize Kalman Filter parameters
        Q = np.eye(3) * 0.01  # Process noise covariance
        R = np.eye(3) * 0.1   # Observation noise covariance
        P = np.eye(3) * 1000  # Initial estimate error covariance
        x = np.zeros((3, 1))  # Initial state estimate (position)
        self.kf = KalmanFilter3DPositionOnly(Q, R, P, x)
        
        self.last_timestamp = self.get_clock().now()

    def uwb_range_callback(self, msg):
        distances = {
            "388" : 0,
            "644" : 0,
            "900" : 0,
            "1156" : 0
        }
        try:
            for anchor_idx in range(MAX_NUM_ANCHORS):
                # distances.append(msg.range_values[index])
                distances[str(msg.anchor_ids[anchor_idx])] = msg.range_values[msg.anchor_ids.index(msg.anchor_ids[anchor_idx])]
        except ValueError as e:
            self.get_logger().error(f"Anchor ID not found: {e}")
            return
        
        # reorder
        

        estimated_position = multilateration(list(distances.values()))
        # self.get_logger().info(f"Estimated position: {estimated_position}")

        # Publish the raw position
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'uwb_tag_link'
        point_msg.point.x = estimated_position[0]
        point_msg.point.y = estimated_position[1]
        point_msg.point.z = estimated_position[2]
        self.uwb_point_raw_pub.publish(point_msg)

        # Kalman Filter update
        current_timestamp = self.get_clock().now()
        dt = (current_timestamp - self.last_timestamp).nanoseconds / 1e9  # Convert to seconds
        self.last_timestamp = current_timestamp

        z = np.array(estimated_position).reshape((3, 1))
        self.kf.predict(dt)
        filtered_position = self.kf.update(z)

        # Publish the filtered position
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'uwb_tag_link'
        point_msg.point.x = filtered_position[0, 0]
        point_msg.point.y = filtered_position[1, 0]
        point_msg.point.z = filtered_position[2, 0]
        self.uwb_point_ekf_pub.publish(point_msg)

        # Publish the updated state as Odometry message
        # odom_msg = Odometry()
        # odom_msg.header.stamp = self.get_clock().now().to_msg()
        # odom_msg.header.frame_id = 'uwb_tag_link'
        # odom_msg.pose.pose.position.x = filtered_position[0, 0]
        # odom_msg.pose.pose.position.y = filtered_position[1, 0]
        # odom_msg.pose.pose.position.z = filtered_position[2, 0]
        # self.uwb_odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    uwb_tag_localizer_node = UwbTagLocalizer()

    rclpy.spin(uwb_tag_localizer_node)
    uwb_tag_localizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
