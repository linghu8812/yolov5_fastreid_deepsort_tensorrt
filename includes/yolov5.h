#ifndef TRACKER_YOLOV5_H
#define TRACKER_YOLOV5_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "model.h"
#include "yaml-cpp/yaml.h"

class YOLOv5 : public Model
{
public:
    explicit YOLOv5(const YAML::Node &yolov5_config);
    ~YOLOv5();
    std::vector<std::vector<DetectRes>> InferenceImages(std::vector<cv::Mat> &vec_img);
    void DrawResults(const std::vector<std::vector <DetectRes>> &detections, std::vector<cv::Mat> &vec_img);

private:
    std::vector<float> prepareImage(std::vector<cv::Mat> &vec_img) override;
    float *ModelInference(std::vector<float> image_data) override;
    std::vector<std::vector<DetectRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
    void NmsDetect(std::vector <DetectRes> &detections);
    static float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);
    std::map<int, std::string> class_labels;
    float obj_threshold;
    float nms_threshold;
    bool agnostic;
    std::vector<int> strides;
    std::vector<std::vector<int>> anchors;
    std::vector<std::vector<int>> grids;
    std::vector<cv::Scalar> class_colors;
};

#endif //TRACKER_YOLOV5_H
