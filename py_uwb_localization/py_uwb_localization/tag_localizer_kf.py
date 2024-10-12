import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from uwb_interfaces.msg import UwbRange
from geometry_msgs.msg import PointStamped

import numpy as np
from scipy.optimize import least_squares
import yaml
import os

def multilateration(anchors, distances):
    
    # Initial guess
    # initial_guess = [0.0, 0.0, 0.0]
    
    # Compute the centroid of the anchors as the initial guess
    initial_guess = np.mean(anchors, axis=0)  # Added this line
    
    
    # Function to minimize
    def residuals(variables):
        x, y, z = variables
        res = [
            np.sqrt((x - anchors[i][0])**2 + (y - anchors[i][1])**2 + (z - anchors[i][2])**2) - distances[i]
            for i in range(len(anchors))
        ]
        return res



    # Solve the system of equations
    solution = least_squares(residuals, initial_guess)
    position = solution.x

    return position

class KalmanFilter3D:
    def __init__(self, Q, R, P, x):
        self.Q = Q  # Process noise covariance (6x6)
        self.R = R  # Observation noise covariance (3x3)
        self.P = P  # Estimate error covariance (6x6)
        self.x = x  # Initial state estimate (6x1)

    def predict(self, dt):
        # State transition matrix
        F = np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1]
        ])

        # Predict the state and estimate error covariance
        self.x = np.dot(F, self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z):
        # Observation matrix
        H = np.array([
            [1, 0, 0, 0, 0, 0],  # Only position is observed
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Compute the Kalman Gain
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update the state estimate and error covariance
        y = z - np.dot(H, self.x)  # Innovation or residual
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, H), self.P)

class UwbTagLocalizer(Node):
    def __init__(self):
        super().__init__('uwb_tag_localizer')
        self.get_logger().info('UwbTagLocalizer node started')

        # Declare parameter for the anchor positions YAML file
        self.declare_parameter('anchor_positions_file', 'anchor_positions.yaml')
        anchor_positions_file = self.get_parameter('anchor_positions_file').value

        # Load anchor positions from the YAML file
        try:
            with open(anchor_positions_file, 'r') as file:
                anchor_positions_data = yaml.safe_load(file)
        except FileNotFoundError:
            self.get_logger().error(f'Anchor positions YAML file not found: {anchor_positions_file}')
            return
        except yaml.YAMLError as exc:
            self.get_logger().error(f'Error parsing anchor positions YAML file: {exc}')
            return

        # Get anchor_positions from the YAML data
        anchor_positions_param = anchor_positions_data.get('anchor_positions')

        if not anchor_positions_param:
            self.get_logger().error('Anchor positions must be provided in the YAML file.')
            return

        # Convert anchor positions to numpy arrays
        self.anchor_positions = {
            str(aid): np.array(pos) for aid, pos in anchor_positions_param.items()
        }
        self.anchor_id_list = list(self.anchor_positions.keys())

        # **Print the anchor positions after reading**
        self.get_logger().info('Anchor positions loaded:')
        for aid, position in self.anchor_positions.items():
            self.get_logger().info(f"  Anchor ID {aid}: Position {position}")

        # Set QoS profile
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
            'uwb_tag_point/kf',
            10
        )

        # Initialize Kalman Filter parameters
        Q = np.eye(6) * 0.01  # Process noise covariance
        R = np.eye(3) * 0.1   # Observation noise covariance
        P = np.eye(6) * 1000  # Initial estimate error covariance
        x = np.zeros((6, 1))  # Initial state estimate (position and velocity)
        self.kf = KalmanFilter3D(Q, R, P, x)

        self.last_timestamp = self.get_clock().now()

    def uwb_range_callback(self, msg):
        # Extract distances corresponding to anchor IDs
        distances = {}
        for anchor_id, range_value in zip(msg.anchor_ids, msg.range_values):
            distances[str(anchor_id)] = range_value

        # Ensure all distances are available
        distances_list = []
        anchors_list = []
        for aid in self.anchor_id_list:
            if aid in distances:
                distances_list.append(distances[aid])
                anchors_list.append(self.anchor_positions[aid])
            else:
                self.get_logger().warning(f"Missing distance for anchor ID {aid}")
                return  # Skip this measurement if any distance is missing

        # Convert lists to numpy arrays
        distances_array = np.array(distances_list)
        anchors_array = np.array(anchors_list)

        # Perform multilateration to estimate position
        estimated_position = multilateration(anchors_array, distances_array)
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

        z = estimated_position.reshape((3, 1))
        self.kf.predict(dt)
        self.kf.update(z)
        filtered_position = self.kf.x[:3].flatten()  # Extract position from state vector

        # Publish the filtered position
        point_msg = PointStamped()
        point_msg.header.stamp = current_timestamp.to_msg()
        point_msg.header.frame_id = 'uwb_tag_link'
        point_msg.point.x = filtered_position[0]
        point_msg.point.y = filtered_position[1]
        point_msg.point.z = filtered_position[2]
        self.uwb_point_ekf_pub.publish(point_msg)

def main(args=None):
    rclpy.init(args=args)
    uwb_tag_localizer_node = UwbTagLocalizer()

    rclpy.spin(uwb_tag_localizer_node)
    uwb_tag_localizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
