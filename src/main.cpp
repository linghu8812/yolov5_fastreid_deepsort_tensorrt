//
// Created by linghu8812 on 2021/10/29.
//

#include <iostream>
#include "yaml-cpp/yaml.h"
#include "video.h"

int main(int argc, char **argv) {
    if (argc < 3)
    {
        std::cout << "Please design config file and video name!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string video_name = argv[2];
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node yolov5_config = root["yolov5"];
    YAML::Node tracker_config = root["tracker"];
    YAML::Node fastreid_config = root["fastreid"];
    YOLOv5 yolov5(yolov5_config);
    yolov5.LoadEngine();
    ObjectTracker tracker(tracker_config);
    fastreid fastreid(fastreid_config);
    fastreid.LoadEngine();
    InferenceVideo(video_name, yolov5, tracker, fastreid);
    return  0;
}