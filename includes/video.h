//
// Created by linghu8812 on 2021/10/29.
//

#ifndef OBJECT_TRACKER_VIDEO_H
#define OBJECT_TRACKER_VIDEO_H

#include "yolov5.h"
#include "tracker.h"
#include "fast-reid.h"

void InferenceVideo(const std::string &video_name, YOLOv5 &yolov5, ObjectTracker &tracker, fastreid &fastreid);

#endif //OBJECT_TRACKER_VIDEO_H
