from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('py_uwb_localization')

    # Path to the RViz configuration file
    rviz_config_file = os.path.join(
        package_dir,
        'rviz',
        'uwb_tag.rviz'  # Ensure this file exists in the specified directory
    )

    # Path to the ground truth YAML file
    yaml_file_path = os.path.join(
        package_dir,
        'config',
        'ground_truth.yaml'  # Ensure this file exists in the specified directory
    )

    # Path to the anchor positions YAML file
    anchor_positions_file = os.path.join(package_dir, 'config', 'anchor_positions.yaml')

    # Node to launch the UWB Tag Localizer
    tag_localizer_node = Node(
        package='py_uwb_localization',
        executable='uwb_tag_localizer',
        name='uwb_tag_localizer',
        output='screen',
        parameters=[{'anchor_positions_file': anchor_positions_file}]
    )

    # Node to launch RViz2 with the specified configuration
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    # Node to launch the Error Evaluation node
    error_evaluation_node = Node(
        package='py_uwb_localization',
        executable='error_evaluation_node',
        name='error_evaluation',
        output='screen',
        parameters=[{'ground_truth_file': yaml_file_path}]
    )
    
    # Node to launch the Micro-ROS Agent
    uros_agent_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=["udp4", "-p", "8888", "-v6"]
    )
    
    # Create the launch description and populate
    ld = LaunchDescription()

    # Add the UWB Tag Localizer node to the launch description
    ld.add_action(tag_localizer_node)

    # Add the RViz node to the launch description
    ld.add_action(rviz_node)

    # Add the Error Evaluation node to the launch description
    ld.add_action(error_evaluation_node)
    
    # Add the Micro-ROS Agent node to the launch description
    ld.add_action(uros_agent_node)

    return ld
