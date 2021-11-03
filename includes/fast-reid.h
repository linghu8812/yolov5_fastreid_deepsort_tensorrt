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
    std::vector<cv::Mat> InferenceImages(std::vector<cv::Mat> &vec_img);
    std::vector<std::vector<cv::Mat>> InferenceImages(const std::vector<cv::Mat> &vec_img,
            const std::vector<std::vector <DetectRes>> &detections);

private:
    std::vector<float> prepareImage(std::vector<cv::Mat> &image) override;
    static std::vector<cv::Mat> CropSubImages(const cv::Mat &org_img, const std::vector<DetectRes> &detection);
    float *ModelInference(std::vector<float> image_data) override;
    static void ReshapeandNormalize(float out[], std::vector<cv::Mat> &feature, const int &MAT_SIZE, const int &outSize);
};

#endif //TRACKER_FAST_REID_H
