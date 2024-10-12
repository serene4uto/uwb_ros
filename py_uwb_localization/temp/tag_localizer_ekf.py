import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.node import Node
from uwb_interfaces.msg import UwbRange
from geometry_msgs.msg import PointStamped

import numpy as np

MAX_NUM_ANCHORS = 4

class KalmanFilterEKF:
    def __init__(self, Q, R, P, x):
        self.Q = Q  # Process noise covariance (6x6)
        self.R = R  # Measurement noise covariance (MxM)
        self.P = P  # Estimate error covariance (6x6)
        self.x = x  # State estimate (6x1)

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
        self.P = np.dot(F, np.dot(self.P, F.T)) + self.Q

    def update(self, z, anchors):
        # Number of measurements
        m = len(z)

        # Measurement prediction
        h = np.zeros((m, 1))
        H = np.zeros((m, 6))

        # Compute h(x) and H matrix
        for i, anchor_pos in enumerate(anchors):
            dx = self.x[0, 0] - anchor_pos[0]
            dy = self.x[1, 0] - anchor_pos[1]
            dz = self.x[2, 0] - anchor_pos[2]
            dist_pred = np.sqrt(dx**2 + dy**2 + dz**2)
            h[i, 0] = dist_pred

            # Compute Jacobian matrix H
            if dist_pred == 0:
                # Avoid division by zero
                H[i, 0:3] = 0
            else:
                H[i, 0] = dx / dist_pred
                H[i, 1] = dy / dist_pred
                H[i, 2] = dz / dist_pred
            # Velocity components have zero partial derivatives
            H[i, 3:6] = 0

        # Innovation or measurement residual
        y = z - h

        # Innovation covariance
        S = np.dot(H, np.dot(self.P, H.T)) + self.R

        # Compute Kalman Gain
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))

        # Update state estimate and covariance matrix
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H, self.P))

class UwbTagLocalizer(Node):
    def __init__(self):
        super().__init__('uwb_tag_localizer')
        self.get_logger().info('UwbTagLocalizer node started')

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

        self.uwb_point_ekf_pub = self.create_publisher(
            PointStamped,
            'uwb_tag_point/ekf',
            10
        )

        # Initialize EKF parameters
        Q = np.eye(6) * 0.1  # Process noise covariance
        R = np.eye(MAX_NUM_ANCHORS) * 0.1   # Measurement noise covariance
        P = np.eye(6) * 1000  # Initial estimate error covariance
        x = np.zeros((6, 1))  # Initial state estimate (position and velocity)
        self.ekf = KalmanFilterEKF(Q, R, P, x)

        self.last_timestamp = self.get_clock().now()

        # Define anchor positions and IDs
        self.anchor_positions = {
            "388": np.array([0.0, 0.0, 0.94]),
            "644": np.array([1.683, 0.0, 1.75]),
            "900": np.array([1.683, 1.205, 2.06]),
            "1156": np.array([0.0, 1.205, 1.33])
        }
        self.anchor_id_list = ["388", "644", "900", "1156"]

    def uwb_range_callback(self, msg):
        # Extract distances corresponding to anchor IDs
        distances = {}
        for anchor_id, range_value in zip(msg.anchor_ids, msg.range_values):
            distances[str(anchor_id)] = range_value

        # Ensure all distances are available
        z_list = []
        anchors_list = []
        for aid in self.anchor_id_list:
            if aid in distances:
                z_list.append(distances[aid])
                anchors_list.append(self.anchor_positions[aid])
            else:
                self.get_logger().warning(f"Missing distance for anchor ID {aid}")
                return  # Skip this measurement if any distance is missing

        # Convert lists to numpy arrays
        z = np.array(z_list).reshape((-1, 1))  # Measurement vector
        anchors = np.array(anchors_list)

        # EKF Prediction
        current_timestamp = self.get_clock().now()
        dt = (current_timestamp - self.last_timestamp).nanoseconds / 1e9  # Convert to seconds
        self.last_timestamp = current_timestamp

        self.ekf.predict(dt)
        self.ekf.update(z, anchors)

        # Extract position estimate
        filtered_position = self.ekf.x[:3].flatten()  # Extract position from state vector

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
