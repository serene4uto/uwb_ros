from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
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

    declare_use_error_eval_arg = DeclareLaunchArgument(
        'use_error_eval',
        default_value='false',
        description='Whether to use the error evaluation node'
    )

    declare_use_uros_arg = DeclareLaunchArgument(
        'use_uros',
        default_value='false',
        description='Whether to use the Micro-ROS Agent'
    )

    declare_use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to use RViz'
    )

    use_error_eval = LaunchConfiguration('use_error_eval')
    use_uros = LaunchConfiguration('use_uros')
    use_rviz = LaunchConfiguration('use_rviz')


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
        arguments=['-d', rviz_config_file],
        condition=IfCondition(use_rviz)
    )

    # Node to launch the Error Evaluation node
    error_evaluation_node = Node(
        package='py_uwb_localization',
        executable='error_evaluation_node',
        name='error_evaluation',
        output='screen',
        parameters=[{'ground_truth_file': yaml_file_path}],
        condition=IfCondition(use_error_eval)
    )
    
    # Node to launch the Micro-ROS Agent
    uros_agent_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=["udp4", "-p", "8888", "-v6"],
        output='screen',
        condition=IfCondition(use_uros)
    )
    

    return LaunchDescription([
        declare_use_error_eval_arg,
        declare_use_uros_arg,
        declare_use_rviz_arg,
        
        tag_localizer_node,
        rviz_node,
        error_evaluation_node,
        uros_agent_node
    ])
