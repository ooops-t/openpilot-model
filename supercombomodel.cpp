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
        Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "supercombo");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

        session = std::make_unique<Ort::Session>(env, path.c_str(), sessionOptions);
        ioBinding = std::make_unique<Ort::IoBinding>(*session);
    }
    
    size_t inputCount = session->GetInputCount();
    size_t outputCount = session->GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    std::cout << "Input count: " <<  inputCount << std::endl;
    for (size_t index = 0; index < inputCount; ++index) {
        auto inputName = session->GetInputNameAllocated(index, allocator);
        std::cout << "Name: " << inputName.get();

        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(index);
        auto typeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = typeAndShapeInfo.GetElementType();
        std::cout << " Type: " << type;

        std::vector<int64_t> shape = typeAndShapeInfo.GetShape();
        std::cout << " Shape: " << shape << std::endl;

        Ort::Value value = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type);
        inputs[inputName.get()] = value.GetTensorMutableRawData();
        ioBinding->BindInput(inputName.get(), std::move(value));
    }

    std::cout << "Output count: " <<  outputCount << std::endl;
    for (size_t index = 0; index < outputCount; ++index) {
        auto outputName = session->GetOutputNameAllocated(index, allocator);
        std::cout << "Name: " << outputName.get();

        Ort::TypeInfo typeInfo = session->GetOutputTypeInfo(index);
        auto typeAndShapeInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = typeAndShapeInfo.GetElementType();
        std::cout << " Type: " << type;

        std::vector<int64_t> shape = typeAndShapeInfo.GetShape();
        std::cout << " Shape: " << shape << std::endl;

        Ort::Value value = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type);
        outputs[outputName.get()] = value.GetTensorMutableRawData();
        ioBinding->BindOutput(outputName.get(), std::move(Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type)));
    }
}

SupercomboModel::~SupercomboModel() {
}

void SupercomboModel::AddInput(const char* name, void* data, ssize_t dataLen) {
    Ort::AllocatorWithDefaultOptions allocator;

    auto input = inputs.find(name);
    if (input == inputs.end()) {
        // not found
        std::cout << "not found" << std::endl; 
    } else {
        memcpy(input->second, data, dataLen);
    }

}

void SupercomboModel::Run() {
    Ort::RunOptions runOptions;

    session->Run(runOptions, *ioBinding);
    std::vector<Ort::Value> outputs = ioBinding->GetOutputValues();
    for (auto itr = outputs.begin(); itr != outputs.end(); ++itr) {
        Ort::Value& value = *itr;
        std::cout << value.GetTensorTypeAndShapeInfo().GetShape() << std::endl;
        const void* data = value.GetTensorRawData();
        for (int i = 0; i < 6504; ++i) {
            printf("%f ", ((const float *)data)[i]);
        }
    }
}
