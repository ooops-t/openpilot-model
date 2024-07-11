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
    std::map<std::string, std::vector<int64_t>> inputs;
    std::map<std::string, std::vector<int64_t>> outputs;
public:
    SupercomboModel() = delete;
    explicit SupercomboModel(std::string path);
    ~SupercomboModel();
};