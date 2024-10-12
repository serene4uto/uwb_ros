import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import yaml
import math
from rclpy.parameter import Parameter

class ErrorEvaluationNode(Node):

    def __init__(self):
        super().__init__('error_evaluation_node')
        
        # Load the ground truth point from YAML
        self.declare_parameter('ground_truth_file', 'ground_truth.yaml')
        ground_truth_file = self.get_parameter('ground_truth_file').value
        
        with open(ground_truth_file, 'r') as file:
            ground_truth_data = yaml.safe_load(file)
        
        self.ground_truth = ground_truth_data['ground_truth']
        
        # Subscriber to the /uwb_tag_point/kf topic
        self.subscription = self.create_subscription(
            PointStamped,
            '/uwb_tag_point/kf',
            self.point_callback,
            10
        )
        
    def point_callback(self, msg):
        # Extract the estimated point from the message
        estimated_point = msg.point
        x_est = estimated_point.x
        y_est = estimated_point.y
        z_est = estimated_point.z
        
        # Extract the ground truth point
        x_gt = self.ground_truth['x']
        y_gt = self.ground_truth['y']
        z_gt = self.ground_truth['z']
        
        # Calculate the error in each axis
        x_error = x_gt - x_est
        y_error = y_gt - y_est
        z_error = z_gt - z_est
        
        # Calculate the 2D error (on the XY plane)
        error_2d = math.sqrt(x_error**2 + y_error**2)
        
        # Calculate the total 3D Euclidean distance error
        error_3d = math.sqrt(x_error**2 + y_error**2 + z_error**2)
        
        # Log the 2D and 3D errors
        self.get_logger().info(f'2D Error (XY plane): {error_2d:.4f}')
        self.get_logger().info(f'3D Error (XYZ): {error_3d:.4f}')
        
        # Log the individual axis errors as well
        self.get_logger().info(f'X error: {x_error:.4f}, Y error: {y_error:.4f}, Z error: {z_error:.4f}')

def main(args=None):
    rclpy.init(args=args)
    
    # Create the node
    node = ErrorEvaluationNode()
    
    # Spin to keep the node running
    rclpy.spin(node)
    
    # Shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
