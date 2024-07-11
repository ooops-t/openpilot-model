#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "supercombomodel.h"

const char *model_path = "supercombo.onnx";
const char *video_path = "video.hevc";


int main(int argc, char *argv[])
{
    SupercomboModel model(model_path);

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