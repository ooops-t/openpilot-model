#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <onnxruntime_cxx_api.h>

const char *model_path = "supercombo.onnx";
const char *video_path = "video.hevc";

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

int main(int argc, char *argv[])
{
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

    Ort::Session session = Ort::Session(env, model_path, sessionOptions);

    size_t inputCount = session.GetInputCount();
    size_t outputCount = session.GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    std::cout << "Input count: " <<  inputCount << std::endl;
    for (int index = 0; index < inputCount; ++index) {
        auto inputName = session.GetInputNameAllocated(index, allocator);
        std::cout << "Name: " << inputName.get();

        Ort::TypeInfo typeInfo = session.GetInputTypeInfo(index);
        auto typeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = typeAndShapeInfo.GetElementType();
        std::cout << " Type: " << type;

        std::vector<int64_t> shape = typeAndShapeInfo.GetShape();
        std::cout << " Shape: " << shape << std::endl;
    }

    std::cout << "Output count: " <<  outputCount << std::endl;
    for (int index = 0; index < outputCount; ++index) {
        auto inputName = session.GetOutputNameAllocated(index, allocator);
        std::cout << "Name: " << inputName.get();

        Ort::TypeInfo typeInfo = session.GetOutputTypeInfo(index);
        auto typeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = typeAndShapeInfo.GetElementType();
        std::cout << " Type: " << type;

        std::vector<int64_t> shape = typeAndShapeInfo.GetShape();
        std::cout << " Shape: " << shape << std::endl;
    }

    // read video
    cv::VideoCapture cap = cv::VideoCapture(video_path);
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open video file" << std::endl;
        return -1; 
    }

    cv::Mat frame;
    while(cv::waitKey(5) <= 0) {
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed" << std::endl;
            break;
        }

        cv::imshow("Live", frame);
    }

    return 0;
}