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

        inputs[inputName.get()] = std::make_pair(shape, type);
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

        outputs[outputName.get()] = std::make_pair(shape, type);
    }
}

SupercomboModel::~SupercomboModel() {
    if (session) {
        delete session;
        session = nullptr;
    }
}

void SupercomboModel::AddInput(const char* name, void* data, ssize_t dataLen) {
    auto input = inputs.find(name);
    if (input == inputs.end()) {
        // not found
        std::cout << "not found" << std::endl; 
    } else {
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<int64_t> shape = input->second.first;
        ONNXTensorElementDataType type = input->second.second;
        Ort::Value value = Ort::Value::CreateTensor(memoryInfo, data, dataLen, shape.data(), shape.size(), type);
        inputNames.emplace_back(name);
        inputValues.emplace_back(std::move(value));
    }

}

void SupercomboModel::AddOutput(const char* name, void *data, ssize_t dataLen) {
    auto output = outputs.find(name);
    if (output == outputs.end()) {
        // not found
        std::cout << "not found" << std::endl; 
    } else {
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<int64_t> shape = output->second.first;
        ONNXTensorElementDataType type = output->second.second;
        Ort::Value value = Ort::Value::CreateTensor(memoryInfo, data, dataLen, shape.data(), shape.size(), type);
        outputNames.emplace_back(name);
        outputValues.emplace_back(std::move(value));
    }
}

void SupercomboModel::Run() {
    Ort::RunOptions runOptions;

    std::vector<Ort::Value> outputs = session->Run(runOptions, inputNames.data(), inputValues.data(), inputValues.size(), outputNames.data(), outputValues.size());
}
