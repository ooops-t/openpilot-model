#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class SupercomboModel {
private:
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::IoBinding> ioBinding;
    std::map<std::string, void*> inputs;
    std::map<std::string, void*> outputs;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
public:
    SupercomboModel() = delete;
    explicit SupercomboModel(std::string path);
    ~SupercomboModel();
    void AddInput(const char* name, void* data, ssize_t dataLen);
    void Run();
};