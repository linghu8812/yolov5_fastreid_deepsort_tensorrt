//
// Created by linghu8812 on 2021/2/8.
//

#ifndef TENSORRT_INFERENCE_MODEL_H
#define TENSORRT_INFERENCE_MODEL_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "common.h"

struct ClassRes{
    int classes;
    float prob;
};

struct DetectRes : ClassRes{
    float x;
    float y;
    float w;
    float h;
};

class Model
{
public:
    void LoadEngine();
    virtual std::vector<float> prepareImage(std::vector<cv::Mat> &image) = 0;
//    virtual float *InferenceImage(std::vector<float> image_data) = 0;
//    virtual bool InferenceFolder(const std::string &folder_name) = 0;

protected:
    bool readTrtFile();
    void onnxToTRTModel();
    virtual float *ModelInference(std::vector<float> image_data) = 0;
    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int CATEGORY;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    cudaStream_t stream;
    int outSize;
    std::vector<float> img_mean;
    std::vector<float> img_std;
};

#endif //TENSORRT_INFERENCE_MODEL_H
