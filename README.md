# Run openpilot supercombo model use ONNX

## Download ONNXRuntime library
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
tar -xvf onnxruntime-linux-x64-gpu-1.18.1.tgz
```

## Build
``` bash
clang++ main.cpp -Ionnxruntime-linux-x64-gpu-1.18.1/include -Lonnxruntime-linux-x64-gpu-1.18.1/lib -lonnxruntime -lopencv_core -lopencv_videoio -lopencv_highgui
```