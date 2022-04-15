#include "SegNet_openvino.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace SegNet

{
    SegNet_OpenVINO::SegNet_OpenVINO()
    {
        std::string projectPath = std::getenv("ROOT_PATH");
        std::string dictPath = projectPath + "/models/model_dict.txt";

        std::fstream in(dictPath);
        if (!in.is_open())
        {
            std::cout << "[ ERROR ] OpenVINO Failed to open the dict path, dictPath = " << dictPath << std::endl;
            std::__throw_runtime_error("[ ERROR ] OpenVINO Failed to open the dict path");
        }
        std::string line, line_modelName, line_modelPath, str_img_h, str_img_w, str_img_c;
        while(std::getline(in, line))
        {
            std::stringstream ss(line);
            ss >> line_modelName >> line_modelPath >> str_img_h >> str_img_w >> str_img_c;
            modelPathDict[line_modelName] = line_modelPath;
            modelImageSizeDict[line_modelName] = std::make_tuple(std::stoi(str_img_h), std::stoi(str_img_w), std::stoi(str_img_c));
        }
        
        availableDevices = ie.GetAvailableDevices();
        for (int i=0; i<(int)availableDevices.size(); i++)
        {
            std::cout << "[ INFO ] OpenVINO Supported Device " << i << " : " << availableDevices[i].c_str() << std::endl;
        }

        device = availableDevices[defaultDeviceID];

        std::cout << "[ INFO ] OpenVINO Selected Device = " << device << std::endl;

        InferenceEngine::Version versionInfo = ie.GetVersions(device)[device];
        std::cout << "[ INFO ] OpenVINO Build Number: " << versionInfo.buildNumber << std::endl;
        std::cout << "[ INFO ] OpenVINO Ver Description: " << versionInfo.description << std::endl;
    }

    int SegNet_OpenVINO::SetModel(std::string modelName)
    {
        std::string projectPath = std::getenv("ROOT_PATH");
        modelPath = projectPath + "/models/" + modelPathDict[modelName];

        std::tie(model_img_h, model_img_w, model_img_c) = modelImageSizeDict[modelName];
        std::ifstream f(modelPath.c_str());
        if (! f.good())
        {
            std::cout << "[ ERROR ] OpenVINO Failed to load selected model" << std::endl;
            std::cout << "[ ERROR ] OpenVINO exhausted loading Path :" << modelPath << std::endl;
            return -1;
            // pending throw exception
            // depracate try catch model
        }
        std::cout << "Load model : " << modelName << std::endl;

        model = ie.ReadNetwork(modelPath);

        if (model.getInputsInfo().empty())
        {
            std::__throw_runtime_error("Error: Empty model inputs info");
        }
        if (model.getOutputsInfo().empty())
        {
            std::__throw_runtime_error("Error: Empty model outputs info");
        }

        inputName = model.getInputsInfo().begin()->first;
        inputInfo = model.getInputsInfo().begin()->second;

        outputName = model.getOutputsInfo().begin()->first;
        outputInfo = model.getOutputsInfo().begin()->second;
        outputInfo->setPrecision(InferenceEngine::Precision::FP32);
        modelExec = ie.LoadNetwork(model, availableDevices[defaultDeviceID]);
        modelInfer = modelExec.CreateInferRequest();

        return 0;
    }
    
    void SegNet_OpenVINO::cvmat2blob(const cv::Mat& frame,
                                    InferenceEngine::InferRequest& inferRequest,
                                    const std::string& inputName) 
    {
        InferenceEngine::Blob::Ptr frameBlob = inferRequest.GetBlob(inputName);
        InferenceEngine::SizeVector blobSize = frameBlob->getTensorDesc().getDims();
        const size_t width = blobSize[3];
        const size_t height = blobSize[2];
        const size_t channels = blobSize[1];

        const size_t image_size = height * width;

        cv::Mat resized_image(frame);
        if ((int)width != frame.size().width || (int)height != frame.size().height) 
        {
            cv::resize(frame, resized_image, cv::Size(width, height));
        }

        float* data = static_cast<float*>(frameBlob->buffer());
        for (size_t row = 0; row < height; row++) 
        {
            for (size_t col = 0; col < width; col++) 
            {
                for (size_t ch = 0; ch < channels; ch++) 
                {
                    data[image_size * ch + row * height + col] = float(resized_image.at<cv::Vec3f>(row, col)[ch]);
                }
            }
        }
    }

    std::tuple<cv::Mat, cv::Mat> SegNet_OpenVINO::GetDualPredictMaskByImage(const cv::Mat& inputOriginImage)
    {
        cv::Mat processedImg;

        auto t0 = std::chrono::high_resolution_clock::now();

        if (model_img_c == 3)
        {
            cv::cvtColor(inputOriginImage, processedImg, cv::COLOR_GRAY2BGR);
            processedImg.convertTo(processedImg, CV_32FC3);
        }
        else if (model_img_c == 1)
        {
            inputOriginImage.convertTo(processedImg, CV_32FC1);
        }
        else
        {
            std::cout << "[ ERROR ] Invalid value of model_img_c, value = " << model_img_c << std::endl;
            std::__throw_runtime_error("[ ERROR ] Invalid value of model_img_c");
        }

        // // UnitTest
        // std::string projectPath = std::getenv("ROOT_PATH");
        // std::string result_save_path = projectPath + "/UnitTest/testResult/openvino_processedImg.png";
        // cv::imwrite(result_save_path, processedImg);

        // Normalize
        float normalize_mean = 0.5;
        float normalize_std = 0.5;
        processedImg = (processedImg / 255 - normalize_mean) / normalize_std;

        cvmat2blob(processedImg, modelInfer, inputName);

        modelInfer.Infer();

        const InferenceEngine::Blob::Ptr outputBlob = modelInfer.GetBlob(outputName);

        const float* skullDataPtr = static_cast<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>(outputBlob->buffer());

        const float* hcDataPtr = skullDataPtr + model_img_h * model_img_w * 2;
        
        cv::Mat predictedHCMask = cv::Mat::zeros(cv::Size(model_img_h, model_img_w), CV_8U);

        for (int row = 0; row < model_img_h; row++) 
        {
            for (int col = 0; col < model_img_w; col++) 
            {
                const float cOne = hcDataPtr[row * model_img_w + col];
                const float cTwo = hcDataPtr[model_img_h * model_img_w + row * model_img_w + col];

                if (cOne < cTwo) 
                {
                    predictedHCMask.ptr<uchar>(row)[col] = 255;
                }
            }
        }

        cv::Mat predictedSkullMask = cv::Mat::zeros(cv::Size(model_img_h, model_img_w), CV_8U);

        for (int row = 0; row < model_img_h; row++) 
        {
            for (int col = 0; col < model_img_w; col++) 
            {
                const float cOne = skullDataPtr[row * model_img_w + col];
                const float cTwo = skullDataPtr[model_img_h * model_img_w + row * model_img_w + col];

                if (cOne < cTwo) 
                {
                    predictedSkullMask.ptr<uchar>(row)[col] = 255;
                }
            }
        }

        // // UnitTest
        // std::string projectPath = std::getenv("ROOT_PATH");
        // std::string result_save_path = projectPath + "/UnitTest/testResult/openvino_predicted_HC_Mask.png";
        // cv::imwrite(result_save_path, predictedHCMask);

        // result_save_path = projectPath + "/UnitTest/testResult/openvino_predicted_Skull_Mask.png";
        // cv::imwrite(result_save_path, predictedSkullMask);


        auto t1 = std::chrono::high_resolution_clock::now();
        auto timeCost = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "[ INFO ] OpenVINO Inference time cost: " << timeCost * 1000 << " ms" << std::endl;

        std::tuple<cv::Mat, cv::Mat> predictedMasks = std::make_tuple(predictedHCMask, predictedSkullMask);
        return predictedMasks;

    }

    SegNet_OpenVINO::~SegNet_OpenVINO()
    {

    }

}
