#include "inference_engine.hpp"
#include "opencv2/opencv.hpp"


namespace SegNet
{
    class SegNet_OpenVINO
    {
    public:
        SegNet_OpenVINO(); // constructer
        ~SegNet_OpenVINO(); // destructor
        int SetModel(std::string modelName);
        cv::Mat GetPredictMaskByImage(const cv::Mat& inputOriginImage);
        std::tuple<cv::Mat, cv::Mat> GetDualPredictMaskByImage(const cv::Mat& inputOriginImage);

        int defaultDeviceID = 0;
        int batchSize = 1;

    private:
        void cvmat2blob(const cv::Mat& frame, InferenceEngine::InferRequest& inferRequest, const std::string& inputName);

    private:
        std::string modelName;
        std::string modelPath;
        std::string dictPath;
        std::map<std::string, std::string> modelPathDict;
        std::map<std::string, std::tuple<int, int, int>> modelImageSizeDict;

        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork model;
        InferenceEngine::ExecutableNetwork modelExec;
        InferenceEngine::InferRequest modelInfer;

        std::vector<std::string> availableDevices;
        std::string device;
        
        std::string inputName;
        InferenceEngine::InputInfo::Ptr inputInfo;
        std::string outputName;
        InferenceEngine::DataPtr outputInfo;

        int model_img_h = 0, model_img_w = 0, model_img_c = 0;
    };

}
