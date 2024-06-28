cmake_minimum_required(VERSION 3.8)
project(uwb_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(uwb_interfaces REQUIRED)

add_executable(tag_localizer src/tag_localizer_node.cpp)
ament_target_dependencies(tag_localizer 
  rclcpp 
  geometry_msgs 
  uwb_interfaces
)

install(TARGETS
  tag_localizer
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
