import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.node import Node
from uwb_interfaces.msg import UwbRange
from geometry_msgs.msg import PointStamped

import numpy as np
from scipy.optimize import least_squares

MAX_NUM_ANCHORS = 4

def multilateration(anchors, distances):
    # Function to minimize
    def residuals(variables):
        x, y, z = variables
        res = [
            np.sqrt((x - anchors[i][0])**2 + (y - anchors[i][1])**2 + (z - anchors[i][2])**2) - distances[i]
            for i in range(len(anchors))
        ]
        return res

    # Initial guess
    initial_guess = [0.0, 0.0, 0.0]

    # Solve the system of equations
    solution = least_squares(residuals, initial_guess)
    position = solution.x

    return position

class KalmanFilter3D:
    def __init__(self, Q, R, P, x):
        self.Q = Q  # Process noise covariance (6x6)
        self.R = R  # Observation noise covariance (3x3)
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
        self.P = self.P - np.dot(K, np.dot(H, self.P))

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

class KalmanFilterUKF:
    def __init__(self, Q, R, P, x):
        self.Q = Q  # Process noise covariance (6x6)
        self.R = R  # Measurement noise covariance (MxM)
        self.P = P  # Estimate error covariance (6x6)
        self.x = x  # State estimate (6x1)

        self.n = self.x.shape[0]  # Dimension of the state vector
        self.kappa = 0  # Scaling parameter, can be adjusted
        self.alpha = 1e-3  # Spread of sigma points
        self.beta = 2  # Optimal for Gaussian distributions

        # Compute lambda parameter
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # Compute weights for mean and covariance
        self.gamma = np.sqrt(self.n + self.lambda_)
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha ** 2 + self.beta)

    def predict(self, dt):
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # Predict sigma points through the process model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[1]):
            sigma_points_pred[:, i] = self._process_model(sigma_points[:, i], dt)

        # Predict state mean and covariance
        self.x = np.dot(self.Wm, sigma_points_pred.T).reshape(-1, 1)
        self.P = self.Q.copy()
        for i in range(sigma_points_pred.shape[1]):
            diff = (sigma_points_pred[:, i].reshape(-1, 1) - self.x)
            self.P += self.Wc[i] * diff @ diff.T

    def update(self, z, anchors):
        m = len(z)  # Number of measurements

        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # Predict measurements
        Z_sigma = np.zeros((m, sigma_points.shape[1]))
        for i in range(sigma_points.shape[1]):
            Z_sigma[:, i] = self._measurement_model(sigma_points[:, i], anchors).flatten()

        # Predicted measurement mean
        z_pred = np.dot(self.Wm, Z_sigma.T).reshape(-1, 1)

        # Innovation covariance matrix S and cross-covariance matrix T
        S = self.R.copy()
        T = np.zeros((self.n, m))
        for i in range(sigma_points.shape[1]):
            z_diff = (Z_sigma[:, i].reshape(-1, 1) - z_pred)
            x_diff = (sigma_points[:, i].reshape(-1, 1) - self.x)
            S += self.Wc[i] * z_diff @ z_diff.T
            T += self.Wc[i] * x_diff @ z_diff.T

        # Kalman Gain
        K = T @ np.linalg.inv(S)

        # Update state and covariance
        y = z - z_pred  # Measurement residual
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

    def _generate_sigma_points(self, x, P):
        sigma_points = np.zeros((self.n, 2 * self.n + 1))
        sigma_points[:, 0] = x.flatten()
        sqrt_P = np.linalg.cholesky((self.n + self.lambda_) * P)
        for i in range(self.n):
            sigma_points[:, i + 1] = x.flatten() + sqrt_P[:, i]
            sigma_points[:, i + 1 + self.n] = x.flatten() - sqrt_P[:, i]
        return sigma_points

    def _process_model(self, x, dt):
        # Constant velocity model
        F = np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1]
        ])
        return F @ x

    def _measurement_model(self, x, anchors):
        # Nonlinear measurement model: distances to anchors
        distances = np.zeros((len(anchors), 1))
        for i, anchor_pos in enumerate(anchors):
            dx = x[0] - anchor_pos[0]
            dy = x[1] - anchor_pos[1]
            dz = x[2] - anchor_pos[2]
            distances[i, 0] = np.sqrt(dx**2 + dy**2 + dz**2)
        return distances

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

        # Publishers for different estimates
        self.uwb_point_raw_pub = self.create_publisher(
            PointStamped,
            'uwb_tag_point/raw',
            10
        )
        self.uwb_point_kf_pub = self.create_publisher(
            PointStamped,
            'uwb_tag_point/kf',
            10
        )
        self.uwb_point_ekf_pub = self.create_publisher(
            PointStamped,
            'uwb_tag_point/ekf',
            10
        )
        self.uwb_point_ukf_pub = self.create_publisher(
            PointStamped,
            'uwb_tag_point/ukf',
            10
        )

        # Initialize Kalman Filter parameters
        Q_kf = np.eye(6) * 0.1  # Process noise covariance
        R_kf = np.eye(3) * 0.1   # Observation noise covariance
        P_kf = np.eye(6) * 1  # Initial estimate error covariance
        x_kf = np.zeros((6, 1))  # Initial state estimate (position and velocity)
        self.kf = KalmanFilter3D(Q_kf, R_kf, P_kf, x_kf)

        # Initialize EKF parameters
        Q_ekf = np.eye(6) * 0.1  # Process noise covariance
        R_ekf = np.eye(MAX_NUM_ANCHORS) * 0.1   # Measurement noise covariance
        P_ekf = np.eye(6) * 1  # Initial estimate error covariance
        x_ekf = np.zeros((6, 1))  # Initial state estimate (position and velocity)
        self.ekf = KalmanFilterEKF(Q_ekf, R_ekf, P_ekf, x_ekf)

        # Initialize UKF parameters
        Q_ukf = np.eye(6) * 0.1  # Process noise covariance
        R_ukf = np.eye(MAX_NUM_ANCHORS) * 0.1   # Measurement noise covariance
        P_ukf = np.eye(6) * 1  # Initial estimate error covariance
        x_ukf = np.zeros((6, 1))  # Initial state estimate (position and velocity)
        self.ukf = KalmanFilterUKF(Q_ukf, R_ukf, P_ukf, x_ukf)

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
        current_timestamp = self.get_clock().now()
        point_msg.header.stamp = current_timestamp.to_msg()
        point_msg.header.frame_id = 'uwb_tag_link'
        point_msg.point.x = estimated_position[0]
        point_msg.point.y = estimated_position[1]
        point_msg.point.z = estimated_position[2]
        self.uwb_point_raw_pub.publish(point_msg)

        # Time delta
        dt = (current_timestamp - self.last_timestamp).nanoseconds / 1e9  # Convert to seconds
        if dt == 0:
            dt = 1e-3  # Prevent division by zero
        self.last_timestamp = current_timestamp

        # Standard Kalman Filter update
        z_kf = estimated_position.reshape((3, 1))
        self.kf.predict(dt)
        self.kf.update(z_kf)
        filtered_position_kf = self.kf.x[:3].flatten()  # Extract position from state vector

        # Publish the KF position
        point_msg_kf = PointStamped()
        point_msg_kf.header.stamp = current_timestamp.to_msg()
        point_msg_kf.header.frame_id = 'uwb_tag_link'
        point_msg_kf.point.x = filtered_position_kf[0]
        point_msg_kf.point.y = filtered_position_kf[1]
        point_msg_kf.point.z = filtered_position_kf[2]
        self.uwb_point_kf_pub.publish(point_msg_kf)

        # EKF update
        z_ekf = distances_array.reshape((-1, 1))  # Measurement vector
        self.ekf.predict(dt)
        self.ekf.update(z_ekf, anchors_array)
        filtered_position_ekf = self.ekf.x[:3].flatten()  # Extract position from state vector

        # Publish the EKF position
        point_msg_ekf = PointStamped()
        point_msg_ekf.header.stamp = current_timestamp.to_msg()
        point_msg_ekf.header.frame_id = 'uwb_tag_link'
        point_msg_ekf.point.x = filtered_position_ekf[0]
        point_msg_ekf.point.y = filtered_position_ekf[1]
        point_msg_ekf.point.z = filtered_position_ekf[2]
        self.uwb_point_ekf_pub.publish(point_msg_ekf)

        # UKF update
        z_ukf = distances_array.reshape((-1, 1))  # Measurement vector
        self.ukf.predict(dt)
        self.ukf.update(z_ukf, anchors_array)
        filtered_position_ukf = self.ukf.x[:3].flatten()  # Extract position from state vector

        # Publish the UKF position
        point_msg_ukf = PointStamped()
        point_msg_ukf.header.stamp = current_timestamp.to_msg()
        point_msg_ukf.header.frame_id = 'uwb_tag_link'
        point_msg_ukf.point.x = filtered_position_ukf[0]
        point_msg_ukf.point.y = filtered_position_ukf[1]
        point_msg_ukf.point.z = filtered_position_ukf[2]
        self.uwb_point_ukf_pub.publish(point_msg_ukf)

def main(args=None):
    rclpy.init(args=args)
    uwb_tag_localizer_node = UwbTagLocalizer()

    rclpy.spin(uwb_tag_localizer_node)
    uwb_tag_localizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
