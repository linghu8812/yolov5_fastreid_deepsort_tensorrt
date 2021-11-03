#ifndef TRACKER_FAST_REID_H
#define TRACKER_FAST_REID_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "model.h"
#include "yaml-cpp/yaml.h"

class fastreid : public Model {
public:
    explicit fastreid(const YAML::Node &yolov5_config);
    ~fastreid();
    std::vector<float> prepareImage(std::vector<cv::Mat> &image) override;
    float *ModelInference(std::vector<float> image_data) override;

private:
    void ReshapeandNormalize(float out[], cv::Mat &feature, const int &MAT_SIZE, const int &outSize);
};

#endif //TRACKER_FAST_REID_H
