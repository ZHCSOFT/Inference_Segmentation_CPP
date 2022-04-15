#include "SegNet_libtorch.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace SegNet

{
    SegNet_Torch::SegNet_Torch()
    {
        std::cout << "[ INFO ] libTorch Ver: " << TORCH_VERSION << std::endl;

        std::string projectPath = std::getenv("ROOT_PATH");
        std::string dictPath = projectPath + "/models/model_dict.txt";
        std::fstream in(dictPath);
        if (! in.is_open())
        {
            std::cout << "[ ERROR ] Failed to open the dict path, dictPath = " << dictPath << std::endl;
            std::__throw_runtime_error("[ ERROR ] Failed to open the dict path, dictPath");
        }
        std::string line, line_modelName, line_modelPath, str_img_h, str_img_w, str_img_c;
        while(std::getline(in, line)){
            std::stringstream ss(line);
            ss >> line_modelName >> line_modelPath >> str_img_h >> str_img_w >> str_img_c;
            modelPathDict[line_modelName] = line_modelPath;
            modelImageSizeDict[line_modelName] = std::make_tuple(std::stoi(str_img_h), std::stoi(str_img_w), std::stoi(str_img_c));
        }

        if(torch::cuda::is_available())
        {
            deviceType = at::kCUDA;
            std::cout << "[ INFO ] libTorch Using CUDA" << std::endl;
        }
        else
        {
            std::cout << "[ INFO ] libTorch Using CPU" << std::endl;
        }
    }

    int SegNet_Torch::SetModel(std::string modelName)
    {
        std::string projectPath = std::getenv("ROOT_PATH");
        modelPath = projectPath + "/OB_AI/models/" + modelPathDict[modelName];

        std::tie(model_img_h, model_img_w, model_img_c) = modelImageSizeDict[modelName];
        std::ifstream f(modelPath.c_str());
        if (! f.good())
        {
            std::cout << "[ ERROR ] libTorch Failed to load selected model" << std::endl;
            std::__throw_runtime_error("Error: libTorch Failed to load selected model");
        }
        std::cout << "[ INFO ] libTorch Load model : " << modelName << std::endl;
        model = torch::jit::load(modelPath);
        model.to(deviceType);
        return 0;
    }

    cv::Mat SegNet_Torch::GetPredictMaskByImage(const cv::Mat& inputOriginImage)
    {
        cv::Mat processedImg;
        auto t0 = std::chrono::high_resolution_clock::now();

        if (model_img_c == 3)
        {
            cv::cvtColor(inputOriginImage, processedImg, cv::COLOR_GRAY2BGR);
            processedImg.convertTo(processedImg, CV_32FC3, 1.0f/255.0f);
        }
        else if (model_img_c == 1)
        {
            inputOriginImage.convertTo(processedImg, CV_32FC1, 1.0f/255.0f);
        }
        else
        {
            std::cout << "[ ERROR ] Invalid value of model_img_c, value = " << model_img_c << std::endl;
            std::__throw_runtime_error("[ ERROR ] Invalid value of model_img_c");
        }
        
        cv::resize(processedImg, processedImg, cv::Size(model_img_h, model_img_w));
        torch::Tensor tensor_img = torch::from_blob(processedImg.data, {model_img_h, model_img_w, model_img_c}, torch::kFloat).clone();
        tensor_img = tensor_img.permute({2, 0, 1}).unsqueeze(0);
        tensor_img = tensor_img.to(deviceType);
        if (model_img_c == 3)
        {
            tensor_img[0][0] = tensor_img[0][0].sub_(0.5).div_(0.5);
            tensor_img[0][1] = tensor_img[0][1].sub_(0.5).div_(0.5);
            tensor_img[0][2] = tensor_img[0][2].sub_(0.5).div_(0.5);
        }
        else if (model_img_c == 1)
        {
            tensor_img[0][0] = tensor_img[0][0].sub_(0.5).div_(0.5);
        }
        else
        {
            std::cout << "[ ERROR ] Invalid value of model_img_c, value = " << model_img_c << std::endl;
            std::__throw_runtime_error("[ ERROR ] Invalid value of model_img_c");
        }

        at::Tensor resultHC = model.forward({tensor_img}).toTensor().squeeze().detach();
        resultHC = resultHC.permute({1, 2, 0}).to(torch::kCPU);
        resultHC = torch::softmax(resultHC, -1);
        resultHC = torch::argmax(resultHC, -1);
        resultHC = resultHC.mul(255).clamp(0, 255).to(torch::kU8);
        cv::Mat predictedHCMask = cv::Mat(model_img_h, model_img_w, CV_8UC1);
        std::memcpy((void *) predictedHCMask.data, resultHC.data_ptr(), sizeof(torch::kU8) * resultHC.numel());
        cv::threshold(predictedHCMask, predictedHCMask, 128, 255, cv::THRESH_BINARY);

        // // UnitTest
        // std::string projectPath = std::getenv("ROOT_PATH");
        // std::string result_save_path = projectPath + "/UnitTest/testResult/libtorch_predicted_single_HC_Mask.png";
        // cv::imwrite(result_save_path, predictedHCMask);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto timeCost = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "[ INFO ] libtorch Inference time cost: " << timeCost * 1000 << " ms" << std::endl;

        return predictedHCMask;
    }

    std::tuple<cv::Mat, cv::Mat> SegNet_Torch::GetDualPredictMaskByImage(const cv::Mat& inputOriginImage)
    {
        cv::Mat processedImg;

        auto t0 = std::chrono::high_resolution_clock::now();

        if (model_img_c == 3)
        {
            cv::cvtColor(inputOriginImage, processedImg, cv::COLOR_GRAY2BGR);
            processedImg.convertTo(processedImg, CV_32FC3, 1.0f/255.0f);
        }
        else if (model_img_c == 1)
        {
            inputOriginImage.convertTo(processedImg, CV_32FC1, 1.0f/255.0f);
        }
        else
        {
            std::cout << "[ ERROR ] Invalid value of model_img_c, value = " << model_img_c << std::endl;
            std::__throw_runtime_error("[ ERROR ] Invalid value of model_img_c");
        }
        
        cv::resize(processedImg, processedImg, cv::Size(model_img_h, model_img_w));
        torch::Tensor tensor_img = torch::from_blob(processedImg.data, {model_img_h, model_img_w, model_img_c}, torch::kFloat).clone();
        tensor_img = tensor_img.permute({2, 0, 1}).unsqueeze(0);
        tensor_img = tensor_img.to(deviceType);

        if (model_img_c == 3)
        {
            tensor_img[0][0] = tensor_img[0][0].sub_(0.5).div_(0.5);
            tensor_img[0][1] = tensor_img[0][1].sub_(0.5).div_(0.5);
            tensor_img[0][2] = tensor_img[0][2].sub_(0.5).div_(0.5);
        }
        else if (model_img_c == 1)
        {
            tensor_img[0][0] = tensor_img[0][0].sub_(0.5).div_(0.5);
        }
        else
        {
            std::cout << "[ ERROR ] Invalid value of model_img_c, value = " << model_img_c << std::endl;
            std::__throw_runtime_error("[ ERROR ] Invalid value of model_img_c");
        }
        
        torch::IValue results = model.forward({tensor_img});
        auto tensorResults = results.toTuple()->elements();

        at::Tensor resultHC = tensorResults[0].toTensor().squeeze().detach();
        at::Tensor resultSkull = tensorResults[1].toTensor().squeeze().detach();

        resultHC = resultHC.permute({1, 2, 0}).to(torch::kCPU);
        resultHC = torch::softmax(resultHC, -1);
        resultHC = torch::argmax(resultHC, -1);
        resultHC = resultHC.mul(255).clamp(0, 255).to(torch::kU8);

        resultSkull = resultSkull.permute({1, 2, 0}).to(torch::kCPU);
        std::cout << resultSkull.sizes() << std::endl;
        resultSkull = torch::softmax(resultSkull, 2);
        resultSkull = torch::argmax(resultSkull, 2);
        resultSkull = resultSkull.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat predictedHCMask = cv::Mat(model_img_h, model_img_w, CV_8UC1);
        cv::Mat predictedSkullMask = cv::Mat(model_img_h, model_img_w, CV_8UC1);
        std::memcpy((void *) predictedHCMask.data, resultHC.data_ptr(), sizeof(torch::kU8) * resultHC.numel());
        std::memcpy((void *) predictedSkullMask.data, resultSkull.data_ptr(), sizeof(torch::kU8) * resultSkull.numel());
        cv::threshold(predictedHCMask, predictedHCMask, 128, 255, cv::THRESH_BINARY);
        cv::threshold(predictedSkullMask, predictedSkullMask, 128, 255, cv::THRESH_BINARY);
        
        // // UnitTest
        // std::string projectPath = std::getenv("ROOT_PATH");
        // std::string result_save_path = projectPath + "/UnitTest/testResult/libtorch_predicted_HC_Mask.png";
        // cv::imwrite(result_save_path, predictedHCMask);
        // result_save_path = projectPath + "/UnitTest/testResult/libtorch_predicted_Skull_Mask.png";
        // cv::imwrite(result_save_path, predictedSkullMask);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto timeCost = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "[ INFO ] libtorch Inference time cost: " << timeCost * 1000 << " ms" << std::endl;

        std::tuple<cv::Mat, cv::Mat> predictedMasks = std::make_tuple(predictedHCMask, predictedSkullMask);
        return predictedMasks;
    }

    SegNet_Torch::~SegNet_Torch()
    {
        
    }
       
}
