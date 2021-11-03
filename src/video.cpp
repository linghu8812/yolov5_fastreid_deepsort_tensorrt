//
// Created by linghu8812 on 2021/10/29.
//
#include "video.h"
#include <opencv2/opencv.hpp>

void InferenceVideo(const std::string &video_name, YOLOv5 &yolov5, ObjectTracker &tracker, fastreid &fastreid) {
    std::cout << "Processing: " << video_name << std::endl;
    cv::VideoCapture video_cap(video_name);
    cv::Size sSize = cv::Size((int) video_cap.get(cv::CAP_PROP_FRAME_WIDTH),
                              (int) video_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width is: " << sSize.width << ", height is: " << sSize.height << std::endl;
    auto fFps = (float)video_cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter video_writer("result.avi",  cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fFps, sSize);
    cv::Mat src_img;
    while (video_cap.read(src_img)) {
        cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
        std::vector<cv::Mat> vec_org_img;
        vec_org_img.push_back(src_img);
        auto detect_boxes = yolov5.InferenceImages(vec_org_img);
        tracker.update(detect_boxes[0], vec_org_img[0].cols, vec_org_img[0].rows);
        tracker.DrawResults(vec_org_img[0]);
        video_writer.write(vec_org_img[0]);
    }
}