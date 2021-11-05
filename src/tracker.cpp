//
// Created by linghu8812 on 2021/11/2.
//

#include "tracker.h"
#include "Hungarian.h"

ObjectTracker::ObjectTracker(const YAML::Node &config) {
    max_age = config["max_age"].as<int>();
    iou_threshold = config["iou_threshold"].as<float>();
    sim_threshold = config["sim_threshold"].as<float>();
    agnostic = config["agnostic"].as<bool>();
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

void ObjectTracker::Alignment(std::vector<std::vector<double>> mat, std::set<int> &unmatchedDetections,
                              std::set<int> &unmatchedTrajectories, std::vector<cv::Point> &matchedPairs,
                              int det_num, int trk_num, bool b_iou) {
    std::vector<int> assignment;
    HungarianAlgorithm HungAlgo;
    HungAlgo.Solve(mat, assignment);

    std::set<int> allItems;
    std::set<int> matchedItems;

    if (b_iou) {
        std::vector<int> detection_index(unmatchedDetections.size());
        std::vector<int> tracker_index(unmatchedTrajectories.size());
        int idx = 0;
        for (const int &umd:unmatchedDetections) {
            detection_index[idx] = umd;
            idx++;
        }
        idx = 0;
        for (const int &umt:unmatchedTrajectories) {
            tracker_index[idx] = umt;
            idx++;
        }
        unmatchedDetections.clear();
        unmatchedTrajectories.clear();
        if (det_num > trk_num) { //	there are unmatched detections
            for (unsigned int n = 0; n < det_num; n++)
                allItems.insert(detection_index[n]);

            for (unsigned int i = 0; i < trk_num; ++i)
                matchedItems.insert(detection_index[assignment[i]]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (det_num < trk_num) { // there are unmatched trajectory/predictions
            for (unsigned int i = 0; i < trk_num; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(tracker_index[i]);
        }
        for (unsigned int i = 0; i < trk_num; ++i) {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - mat[i][assignment[i]] < iou_threshold) {
                unmatchedTrajectories.insert(tracker_index[i]);
                unmatchedDetections.insert(detection_index[assignment[i]]);
            }
            else
                matchedPairs.emplace_back(cv::Point(tracker_index[i], detection_index[assignment[i]]));
        }
    } else {
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
        for (unsigned int i = 0; i < trk_num; ++i) {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - mat[i][assignment[i]] < sim_threshold) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs.emplace_back(cv::Point(i, assignment[i]));
        }
    }
}

void ObjectTracker::FeatureMatching(const std::vector<TrackerRes> &predict_boxes, std::set<int> &unmatchedDetections,
                                    std::set<int> &unmatchedTrajectories, std::vector<cv::Point> &matchedPairs) {
    int det_num = tracker_boxes.size();
    int trk_num = predict_boxes.size();
    std::vector<std::vector<double>> similar_mat(trk_num, std::vector<double>(det_num, 0));
    for (unsigned int i = 0; i < trk_num; i++) { // compute iou matrix as a distance matrix
        for (unsigned int j = 0; j < det_num; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            if (predict_boxes[i].classes == tracker_boxes[j].classes or agnostic) {
                similar_mat[i][j] = 1 - predict_boxes[i].feature.dot(tracker_boxes[j].feature);
            } else
                similar_mat[i][j] = 1;
        }
    }
    Alignment(similar_mat, unmatchedDetections, unmatchedTrajectories, matchedPairs, det_num, trk_num, false);
}

void ObjectTracker::IOUMatching(const std::vector<TrackerRes> &predict_boxes, std::set<int> &unmatchedDetections,
                                std::set<int> &unmatchedTrajectories, std::vector<cv::Point> &matchedPairs) {
    int det_num = unmatchedDetections.size();
    int trk_num = unmatchedTrajectories.size();
    if (det_num == 0 or trk_num == 0)
        return;
    std::vector<std::vector<double>> iou_mat(trk_num, std::vector<double>(det_num, 0));
    int i = 0;
    for (const int &umt : unmatchedTrajectories) { // compute iou matrix as a distance matrix
        int j = 0;
        for (const int &umd : unmatchedDetections) {
            if (predict_boxes[umt].classes == tracker_boxes[umd].classes or agnostic) {
                iou_mat[i][j] = 1 - IOUCalculate(predict_boxes[umt], tracker_boxes[umd]);
            } else
                iou_mat[i][j] = 1;
            j++;
        }
        i++;
    }
    Alignment(iou_mat, unmatchedDetections, unmatchedTrajectories, matchedPairs, det_num, trk_num, true);
}

void ObjectTracker::update(const std::vector<DetectRes> &det_boxes, const std::vector<cv::Mat> &det_features,
        int width, int height) {
    tracker_boxes.clear();
    int index = 0;
    for (const auto &det_box : det_boxes) {
        TrackerRes tracker_box(det_box.classes, det_box.prob, det_box.x, det_box.y, det_box.w, det_box.h, -1);
        tracker_box.feature = det_features[index];
        index++;
        tracker_boxes.push_back(tracker_box);
    }
    if (kalman_boxes.empty()) {
        for (auto &tracker_box : tracker_boxes) {
            StateType rect_box = { tracker_box.x, tracker_box.y, tracker_box.w, tracker_box.h };
            KalmanTracker tracker = KalmanTracker(rect_box, tracker_box.classes, tracker_box.prob);
            tracker.m_feature = tracker_box.feature.clone();
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
        trk_box.feature = it->m_feature;
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

    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::vector<cv::Point> matchedPairs;
    FeatureMatching(predict_boxes, unmatchedDetections, unmatchedTrajectories, matchedPairs);
    IOUMatching(predict_boxes, unmatchedDetections, unmatchedTrajectories, matchedPairs);

    for (auto & matchedPair : matchedPairs) {
        int trk_id = matchedPair.x;
        int det_id = matchedPair.y;
        StateType rect_box = { tracker_boxes[det_id].x, tracker_boxes[det_id].y,
                               tracker_boxes[det_id].w, tracker_boxes[det_id].h };
        kalman_boxes[trk_id].update(rect_box, tracker_boxes[det_id].classes, tracker_boxes[det_id].prob,
                tracker_boxes[det_id].feature);
        tracker_boxes[det_id].object_id = kalman_boxes[trk_id].m_id;
    }
    for (auto umd : unmatchedDetections) {
        StateType rect_box = { tracker_boxes[umd].x, tracker_boxes[umd].y,
                               tracker_boxes[umd].w, tracker_boxes[umd].h };
        KalmanTracker tracker = KalmanTracker(rect_box, tracker_boxes[umd].classes, tracker_boxes[umd].prob);
        tracker_boxes[umd].object_id = tracker.m_id;
        tracker.m_feature =  tracker_boxes[umd].feature.clone();
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