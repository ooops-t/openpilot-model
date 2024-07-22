#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class SupercomboModel {
private:
    /* data */
    Ort::Session* session;
    std::map<std::string, std::pair<std::vector<int64_t>, ONNXTensorElementDataType>> inputs;
    std::map<std::string, std::pair<std::vector<int64_t>, ONNXTensorElementDataType>> outputs;
    std::vector<const char*> inputNames;
    std::vector<Ort::Value> inputValues;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> outputValues;
public:
    SupercomboModel() = delete;
    explicit SupercomboModel(std::string path);
    ~SupercomboModel();
    void AddInput(const char* name, void* data, ssize_t dataLen);
    void AddOutput(const char* name, void* data, ssize_t dataLen);
    void Run();
};