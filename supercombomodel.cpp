#include "supercombomodel.h"

#include <iostream>

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

SupercomboModel::SupercomboModel(std::string path) : session(nullptr) {
    {
        Ort::Env env;
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

        session = new Ort::Session(env, path.c_str(), sessionOptions);
    }
    
    size_t inputCount = session->GetInputCount();
    size_t outputCount = session->GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    std::cout << "Input count: " <<  inputCount << std::endl;
    for (int index = 0; index < inputCount; ++index) {
        auto inputName = session->GetInputNameAllocated(index, allocator);
        std::cout << "Name: " << inputName.get();

        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(index);
        auto typeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = typeAndShapeInfo.GetElementType();
        std::cout << " Type: " << type;

        std::vector<int64_t> shape = typeAndShapeInfo.GetShape();
        std::cout << " Shape: " << shape << std::endl;

        inputs[inputName.get()] = shape;
    }

    std::cout << "Output count: " <<  outputCount << std::endl;
    for (int index = 0; index < outputCount; ++index) {
        auto outputName = session->GetOutputNameAllocated(index, allocator);
        std::cout << "Name: " << outputName.get();

        Ort::TypeInfo typeInfo = session->GetOutputTypeInfo(index);
        auto typeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = typeAndShapeInfo.GetElementType();
        std::cout << " Type: " << type;

        std::vector<int64_t> shape = typeAndShapeInfo.GetShape();
        std::cout << " Shape: " << shape << std::endl;

        outputs[outputName.get()] = shape;
    }
}

SupercomboModel::~SupercomboModel() {
    if (session) {
        delete session;
        session = nullptr;
    }
}