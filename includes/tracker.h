//
// Created by linghu8812 on 2021/11/2.
//

#ifndef OBJECT_TRACKER_TRACKER_H
#define OBJECT_TRACKER_TRACKER_H

#include "yolov5.h"
#include "KalmanTracker.h"

struct TrackerRes : public DetectRes {
    TrackerRes(int cl, float pb, float xc, float yc, float wc, float hc, int id) : DetectRes() {
        classes = cl, prob = pb, x = xc, y = yc, w = wc, h = hc, object_id = id;
    }
    cv::Mat feature;
    int object_id;
};

class ObjectTracker {
public:
    explicit ObjectTracker(const YAML::Node &config);
    ~ObjectTracker();
    void update(const std::vector<DetectRes> &det_boxes, const std::vector<cv::Mat> &det_features,
            int width, int height);
    void DrawResults(cv::Mat &origin_img);

public:
    std::vector<TrackerRes> tracker_boxes;
private:
    static float IOUCalculate(const TrackerRes &det_a, const TrackerRes &det_b);
    int max_age;
    float iou_threshold;
    float sim_threshold;
    std::vector<KalmanTracker> kalman_boxes;
    std::map<int, std::string> class_labels;
    std::string labels_file;
    std::vector<cv::Scalar> id_colors;
};

#endif //OBJECT_TRACKER_TRACKER_H
