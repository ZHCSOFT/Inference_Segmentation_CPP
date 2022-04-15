#pragma once
#include "torch/torch.h"
#include "torch/script.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include "sys/stat.h"
#include "sys/types.h"

namespace SegNet
{
    class SegNet_Torch
    {
    public:
        SegNet_Torch(); // constructer
        ~SegNet_Torch(); // destructor
        int SetModel(std::string modelName);
        cv::Mat GetPredictMaskByImage(const cv::Mat& inputOriginImage);
        std::tuple<cv::Mat, cv::Mat> GetDualPredictMaskByImage(const cv::Mat& inputOriginImage);

    private:
        std::string modelName;
        std::string modelPath;
        std::string dictPath;
        std::map<std::string, std::string> modelPathDict;
        std::map<std::string, std::tuple<int, int, int>> modelImageSizeDict;
        torch::DeviceType deviceType = at::kCPU;
        torch::jit::script::Module model;
        
        int model_img_h = 0, model_img_w = 0, model_img_c = 0;
    };

}
