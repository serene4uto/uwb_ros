#include "rclcpp/rclcpp.hpp"
#include "uwb_interfaces/msg/uwb_range.hpp"
#include "geometry_msgs/msg/point.hpp"

#define MAX_ANCHORS 4


class TagLocalizerNode : public rclcpp::Node {
public:
    TagLocalizerNode() 
    : Node("tag_localizer_node") {
        RCLCPP_INFO(this->get_logger(), "Tag Localizer Node has been started.");

        uwb_range_sub_ = this->create_subscription<uwb_interfaces::msg::UwbRange>(
            "/uros_esp32_uwb_tag_range", 10, std::bind(&TagLocalizerNode::uwbRangeCallback, this, std::placeholders::_1)
        );

        uwb_tag_position_pub_ = this->create_publisher<geometry_msgs::msg::Point>(
            "/uwb_tag_position", 10
        );
    }

private:
    rclcpp::Subscription<uwb_interfaces::msg::UwbRange>::SharedPtr uwb_range_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr uwb_tag_position_pub_;

    void uwbRangeCallback(const uwb_interfaces::msg::UwbRange::SharedPtr msg) {

    }


};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TagLocalizerNode>());
    rclcpp::shutdown();
    return 0;
}