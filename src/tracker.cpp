//
// Created by linghu8812 on 2021/11/2.
//

#include "tracker.h"
#include "Hungarian.h"

ObjectTracker::ObjectTracker(const YAML::Node &config) {
    max_age = config["max_age"].as<int>();
    iou_threshold = config["iou_threshold"].as<float>();
    labels_file = config["labels_file"].as<std::string>();
    class_labels = readClassLabel(labels_file);
    id_colors.resize(100);
    srand((int) time(nullptr));
    for (cv::Scalar &id_color : id_colors)
        id_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

ObjectTracker::~ObjectTracker() = default;

float ObjectTracker::IOUCalculate(const TrackerRes &det_a, const TrackerRes &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}

void ObjectTracker::update(const std::vector<DetectRes> &det_boxes, int width, int height) {
    tracker_boxes.clear();
    for (const auto &det_box : det_boxes) {
        TrackerRes tracker_box(det_box.classes, det_box.prob, det_box.x, det_box.y, det_box.w, det_box.h, -1);
        tracker_boxes.push_back(tracker_box);
    }
    if (kalman_boxes.empty()) {
        for (auto &tracker_box : tracker_boxes) {
            StateType rect_box = { tracker_box.x, tracker_box.y, tracker_box.w, tracker_box.h };
            KalmanTracker tracker = KalmanTracker(rect_box, tracker_box.classes, tracker_box.prob);
            tracker_box.object_id = tracker.m_id;
            kalman_boxes.push_back(tracker);
        }
        return;
    }
    std::vector<TrackerRes> predict_boxes;
    for (auto it = kalman_boxes.begin(); it != kalman_boxes.end();)
    {
        cv::Rect_<float> pBox = (*it).predict();

        bool is_nan = (isnan(pBox.x)) or (isnan(pBox.y)) or (isnan(pBox.width)) or (isnan(pBox.height));
        bool is_bound = (pBox.x > (float)width) or (pBox.y > (float)height) or
                (pBox.x + pBox.width < 0) or (pBox.y + pBox.height < 0);
        bool is_illegal = (pBox.width <= 0) or (pBox.height <= 0);
        bool time_since_update = it->m_time_since_update > max_age;

        TrackerRes trk_box(it->m_classes, it->m_prob, pBox.x, pBox.y, pBox.width, pBox.height, it->m_id);
        trk_box.classes = it->m_classes;
        if (!(time_since_update or is_nan or is_bound or is_illegal))
        {
            predict_boxes.push_back(trk_box);
            it++;
        }
        else
        {
            it = kalman_boxes.erase(it);
            //cerr << "Box invalid at frame: " << frame_count << endl;
        }
    }
    int det_num = tracker_boxes.size();
    int trk_num = predict_boxes.size();
    std::vector<std::vector<double>> iouMatrix;
    iouMatrix.resize(trk_num, std::vector<double>(det_num, 0));
    for (unsigned int i = 0; i < trk_num; i++) { // compute iou matrix as a distance matrix
        for (unsigned int j = 0; j < det_num; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            if (predict_boxes[i].classes == tracker_boxes[j].classes)
                iouMatrix[i][j] = 1 - IOUCalculate(predict_boxes[i], tracker_boxes[j]);
            else
                iouMatrix[i][j] = 1;
        }
    }
    std::vector<int> assignment;
    HungarianAlgorithm HungAlgo;
    HungAlgo.Solve(iouMatrix, assignment);

    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;

    if (det_num > trk_num) { //	there are unmatched detections
        for (unsigned int n = 0; n < det_num; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trk_num; ++i)
            matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (det_num < trk_num) { // there are unmatched trajectory/predictions
        for (unsigned int i = 0; i < trk_num; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }

    std::vector<cv::Point> matchedPairs;
    for (unsigned int i = 0; i < trk_num; ++i) {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iou_threshold) {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.emplace_back(cv::Point(i, assignment[i]));
    }

    for (auto & matchedPair : matchedPairs) {
        int trk_id = matchedPair.x;
        int det_id = matchedPair.y;
        StateType rect_box = { tracker_boxes[det_id].x, tracker_boxes[det_id].y,
                               tracker_boxes[det_id].w, tracker_boxes[det_id].h };
        kalman_boxes[trk_id].update(rect_box, tracker_boxes[det_id].classes, tracker_boxes[det_id].prob);
        tracker_boxes[det_id].object_id = kalman_boxes[trk_id].m_id;
    }
    for (auto umd : unmatchedDetections) {
        StateType rect_box = { tracker_boxes[umd].x, tracker_boxes[umd].y,
                               tracker_boxes[umd].w, tracker_boxes[umd].h };
        KalmanTracker tracker = KalmanTracker(rect_box, tracker_boxes[umd].classes, tracker_boxes[umd].prob);
        tracker_boxes[umd].object_id = tracker.m_id;
        kalman_boxes.push_back(tracker);
    }
}

void ObjectTracker::DrawResults(cv::Mat &origin_img) {
    cv::cvtColor(origin_img, origin_img, cv::COLOR_BGR2RGB);
    for(const auto &tracker_box : tracker_boxes) {
        char t[256];
        sprintf(t, "%d", tracker_box.object_id);
        std::string name = class_labels[tracker_box.classes] + "-" + t;
        cv::putText(origin_img, name, cv::Point(tracker_box.x - tracker_box.w / 2, tracker_box.y - tracker_box.h / 2 - 5),
                    cv::FONT_HERSHEY_COMPLEX, 0.7, id_colors[tracker_box.object_id % 100], 2);
        cv::Rect rst(tracker_box.x - tracker_box.w / 2, tracker_box.y - tracker_box.h / 2, tracker_box.w, tracker_box.h);
        cv::rectangle(origin_img, rst, id_colors[tracker_box.object_id % 100], 2, cv::LINE_8, 0);
    }
}