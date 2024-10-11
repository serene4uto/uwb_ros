from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    tag_localizer_node = Node(
        package='py_uwb_localization',
        executable='uwb_tag_localizer',
        name='py_tag_localizer',
        output='screen'
    )

    # ekf_filter_node = Node(
    #     package='robot_localization',
    #     executable='ekf_node',
    #     name='ekf_filter_node',
    #     output='screen',
    #     parameters=[os.path.join(get_package_share_directory("py_uwb_localization"), 'config', 'uwb_ekf.yaml')],

    # )
    
    ld = LaunchDescription()

    ld.add_action(tag_localizer_node)
    # ld.add_action(ekf_filter_node)

    return ld
