#include <chrono>
#include <iostream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "supercombomodel.h"

const char *model_path = "supercombo.onnx";
const char *video_path = "video.hevc";

static std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


int main(int argc, char *argv[])
{
    SupercomboModel model(model_path);
#if 1
    // read video
    cv::VideoCapture cap = cv::VideoCapture(video_path, cv::CAP_ANY);
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

        // reize the frame
        cv::Mat nframe;
        cv::resize(frame, nframe, cv::Size(512, 256));
        // std::cout << "Resize rows: " << nframe.rows << " cols: " << nframe.cols << " Channel: " << nframe.channels() << std::endl;

        // convert to YUV420
        cv::Mat yuv;
        cv::cvtColor(nframe, yuv, cv::COLOR_BGR2YUV);

        // split YUV
        std::vector<cv::Mat> channels;
        cv::split(yuv, channels);
    
        cv::imshow("Live Y", channels[0]);
        cv::imshow("Live U", channels[1]);
        cv::imshow("Live V", channels[2]);

        cv::Mat newframe = cv::Mat::zeros(cv::Size(512, 256), CV_8UC(6));

        std::vector<cv::Mat> t;
        cv::split(newframe, t);
        
        cv::resize(channels[1], t[4], cv::Size(256, 128));
        cv::resize(channels[2], t[5], cv::Size(256, 128));
        cv::imshow("Live x", t[4]);

        // 20Hz = 1 / 20 = 0.050s = 50ms
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    cap.release();
#else
    cv::Mat frame = cv::imread("preview.png");
    std::cout << "Depth: " <<  frame.depth() << " type: " << type2str(frame.type()) << std::endl;;
    std::cout << "Origin rows: " << frame.rows << " cols: " << frame.cols << " Channel: " << frame.channels() << std::endl;

    cv::Mat yuv;
    cv::cvtColor(frame, yuv, cv::COLOR_BGR2YUV);

    // split YUV
    std::vector<cv::Mat> channels;
    cv::split(yuv, channels);
    std::cout << "channels rows: " << channels[0].rows << " cols: " << channels[0].cols << " Channel: " << channels[0].channels() << std::endl;
    cv::imshow("Live Y", channels[0]);
    cv::imshow("Live U", channels[1]);
    cv::imshow("Live V", channels[2]);

    while(cv::waitKey(5) <= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
#endif
    return 0;
}