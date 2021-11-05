#include "yolov5.h"
#include "yaml-cpp/yaml.h"

YOLOv5::YOLOv5(const YAML::Node &config) {
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    labels_file = config["labels_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    agnostic = config["agnostic"].as<bool>();
    strides = config["strides"].as<std::vector<int>>();
    anchors = config["anchors"].as<std::vector<std::vector<int>>>();
    class_labels = readClassLabel(labels_file);
    CATEGORY = class_labels.size();
    grids = {
            {3, int(IMAGE_WIDTH / strides[0]), int(IMAGE_HEIGHT / strides[0])},
            {3, int(IMAGE_WIDTH / strides[1]), int(IMAGE_HEIGHT / strides[1])},
            {3, int(IMAGE_WIDTH / strides[2]), int(IMAGE_HEIGHT / strides[2])},
    };
    class_colors.resize(CATEGORY);
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

YOLOv5::~YOLOv5() = default;

std::vector<std::vector<DetectRes>> YOLOv5::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = prepareImage(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "yolov5 prepare image take: " << total_pre << " ms." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    auto *output = ModelInference(image_data);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "yolov5 inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto boxes = postProcess(vec_img, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "yolov5 postprocess take: " << total_res << " ms." << std::endl;
    delete output;
    return boxes;
}

std::vector<float> YOLOv5::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    int index = 0;
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        float ratio = float(IMAGE_WIDTH) / float(src_img.cols) < float(IMAGE_HEIGHT) / float(src_img.rows) ? float(IMAGE_WIDTH) / float(src_img.cols) : float(IMAGE_HEIGHT) / float(src_img.rows);
        cv::Mat flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
        cv::Mat rsz_img;
        cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
        flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);

        //HWC TO CHW
        int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
        std::vector<cv::Mat> split_img = {
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * (index + 2)),
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * (index + 1)),
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * index)
        };
        index += 3;
        cv::split(flt_img, split_img);
    }
    return result;
}

float *YOLOv5::ModelInference(std::vector<float> image_data) {
    auto *out = new float[outSize * BATCH_SIZE];
    if (!image_data.data()) {
        std::cout << "prepare images ERROR!" << std::endl;
        return out;
    }
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    cudaMemcpyAsync(buffers[0], image_data.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

    // do inference
    context->execute(BATCH_SIZE, buffers);
    cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return out;
}

std::vector<std::vector<DetectRes>> YOLOv5::postProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<std::vector<DetectRes>> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        std::vector<DetectRes> result;
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);
        float *out = output + index * outSize;
        int position = 0;
        for (int n = 0; n < (int)grids.size(); n++)
        {
            for (int c = 0; c < grids[n][0]; c++)
            {
                std::vector<int> anchor = anchors[n * grids[n][0] + c];
                for (int h = 0; h < grids[n][1]; h++)
                    for (int w = 0; w < grids[n][2]; w++)
                    {
                        float *row = out + position * (CATEGORY + 5);
                        position++;
                        DetectRes box;
                        auto max_pos = std::max_element(row + 5, row + CATEGORY + 5);
                        box.prob = row[4] * row[max_pos - row];
                        if (box.prob < obj_threshold)
                            continue;
                        box.classes = max_pos - row - 5;
                        box.x = (row[0] * 2 - 0.5 + w) / grids[n][1] * IMAGE_WIDTH * ratio;
                        box.y = (row[1] * 2 - 0.5 + h) / grids[n][2] * IMAGE_HEIGHT * ratio;
                        box.w = pow(row[2] * 2, 2) * anchor[0] * ratio;
                        box.h = pow(row[3] * 2, 2) * anchor[1] * ratio;
                        result.push_back(box);
                    }
            }
        }
        NmsDetect(result);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

void YOLOv5::NmsDetect(std::vector<DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const DetectRes &left, const DetectRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].classes == detections[j].classes or agnostic)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > nms_threshold)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes &det)
    { return det.prob == 0; }), detections.end());
}

float YOLOv5::IOUCalculate(const DetectRes &det_a, const DetectRes &det_b) {
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

void YOLOv5::DrawResults(const std::vector<std::vector<DetectRes>> &detections, std::vector<cv::Mat> &vec_img) {
    for (int i = 0; i < (int)vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;
        auto rects = detections[i];
        cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
        for(const auto &rect : rects) {
            char t[256];
            sprintf(t, "%.2f", rect.prob);
            std::string name = class_labels[rect.classes] + "-" + t;
            cv::putText(org_img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5),
                    cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[rect.classes], 2);
            cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
            cv::rectangle(org_img, rst, class_colors[rect.classes], 2, cv::LINE_8, 0);
        }
    }
}
