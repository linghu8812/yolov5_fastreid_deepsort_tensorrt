#include "fast-reid.h"
#include "yaml-cpp/yaml.h"

fastreid::fastreid(const YAML::Node &config) {
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
}

fastreid::~fastreid() = default;

std::vector<cv::Mat> fastreid::InferenceImages(std::vector<cv::Mat> &vec_img) {
    std::vector<cv::Mat> res_feature;
    int start_index = 0, end_index = 0;
    while (end_index < (int)vec_img.size()) {
        end_index = std::min(int(start_index + BATCH_SIZE), int(vec_img.size()));
        std::vector<cv::Mat>::const_iterator iter_1 = vec_img.begin() + start_index;
        std::vector<cv::Mat>::const_iterator iter_2 = vec_img.begin() + end_index;
        std::vector<cv::Mat> sub_vec_img(iter_1, iter_2);
        auto t_start_pre = std::chrono::high_resolution_clock::now();
        std::vector<float> image_data = prepareImage(sub_vec_img);
        auto t_end_pre = std::chrono::high_resolution_clock::now();
        float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
        std::cout << "fast-reid prepare image take: " << total_pre << " ms." << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        auto *output = ModelInference(image_data);
        auto t_end = std::chrono::high_resolution_clock::now();
        float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        std::cout << "fast-reid inference take: " << total_inf << " ms." << std::endl;
        auto r_start = std::chrono::high_resolution_clock::now();
//        cv::Mat feature(BATCH_SIZE, outSize, CV_32FC1);
        std::vector<cv::Mat> feature;
        ReshapeandNormalize(output, feature, iter_2 - iter_1, outSize);
        res_feature.insert(res_feature.end(), feature.begin(), feature.end());
        auto r_end = std::chrono::high_resolution_clock::now();
        float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
        std::cout << "fast-reid post process take: " << total_res << " ms." << std::endl;
        delete output;
        start_index += BATCH_SIZE;
    }
    return res_feature;
}

std::vector<cv::Mat> fastreid::CropSubImages(const cv::Mat &org_img, const std::vector<DetectRes> &detection) {
    std::vector<cv::Mat> vec_mat;
    for (const auto &bbox : detection) {
        int x1 = std::max(int(bbox.x - bbox.w / 2), 0), y1 = std::max(int(bbox.y - bbox.h / 2), 0),
        x2 = std::min(int(bbox.x + bbox.w / 2), org_img.cols), y2 = std::min(int(bbox.y + bbox.h / 2), org_img.rows);
        cv::Rect rect{x1, y1, x2 - x1, y2 - y1};
        cv::Mat sub_img = org_img(rect);
        vec_mat.push_back(sub_img);
    }
    return vec_mat;
}

std::vector<std::vector<cv::Mat>> fastreid::InferenceImages(const std::vector<cv::Mat> &vec_img,
                                                            const std::vector<std::vector<DetectRes>> &detections) {
    assert(vec_img.size() == detections.size());
    std::vector<std::vector<cv::Mat>> res_feature;
    for (int i = 0; i < (int)vec_img.size(); i++) {
        const cv::Mat &org_img = vec_img[i];
        const std::vector<DetectRes> detection = detections[i];
        auto vec_mat = CropSubImages(org_img, detection);
        auto vec_features = InferenceImages(vec_mat);
        res_feature.push_back(vec_features);
    }
    return res_feature;
}

std::vector<float> fastreid::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        cv::Mat flt_img;
        cv::resize(src_img, flt_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        flt_img.convertTo(flt_img, CV_32FC3);

        //HWC TO CHW
        std::vector<cv::Mat> split_img(INPUT_CHANNEL);
        cv::split(flt_img, split_img);

        int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
        for (int i = 0; i < INPUT_CHANNEL; ++i)
        {
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}

float *fastreid::ModelInference(std::vector<float> image_data) {
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

void fastreid::ReshapeandNormalize(float *out, std::vector<cv::Mat> &feature, const int &MAT_SIZE, const int &outSize) {
    for (int i = 0; i < MAT_SIZE; i++)
    {
        cv::Mat onefeature(1, outSize, CV_32FC1, out + i * outSize);
        cv::normalize(onefeature, onefeature);
        feature.push_back(onefeature.clone());
    }
}
