#include <chrono>
#include <iostream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "supercombomodel.h"

const char *model_path = "supercombo_f32.onnx";
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
    #define input_imgs_size (1*12*128*256)
    #define big_input_imgs_size (1*12*128*256)
    #define desire_size (1*100*8)
    #define traffic_convention_size (1*2)
    #define lateral_control_params_size (1*2)
    #define prev_desired_curv_size (1*100*1)
    #define nav_features_size (1*256)
    #define nav_instructions_size (1*150)
    #define features_buffer_size (1*99*512)

    srand(time(NULL));

    SupercomboModel model(model_path);

    std::array<float, input_imgs_size> input_imgs;
    std::generate(input_imgs.begin(), input_imgs.end(), rand);
    model.AddInput("input_imgs", input_imgs.data(), input_imgs_size);

    std::array<float, big_input_imgs_size> big_input_imgs;
    std::generate(big_input_imgs.begin(), big_input_imgs.end(), rand);
    model.AddInput("big_input_imgs", big_input_imgs.data(), big_input_imgs_size);

    std::array<float, desire_size> desire;
    std::generate(desire.begin(), desire.end(), rand);
    model.AddInput("desire", desire.data(), desire_size);

    std::array<float, traffic_convention_size> traffic_convention;
    std::generate(traffic_convention.begin(), traffic_convention.end(), rand);
    model.AddInput("traffic_convention", traffic_convention.data(), traffic_convention_size);

    std::array<float, lateral_control_params_size> lateral_control_params;
    std::generate(lateral_control_params.begin(), lateral_control_params.end(), rand);
    model.AddInput("lateral_control_params", lateral_control_params.data(), lateral_control_params_size);

    std::array<float, prev_desired_curv_size> prev_desired_curv;
    std::generate(prev_desired_curv.begin(), prev_desired_curv.end(), rand);
    model.AddInput("prev_desired_curv", prev_desired_curv.data(), prev_desired_curv_size);

    std::array<float, nav_features_size> nav_features;
    std::generate(nav_features.begin(), nav_features.end(), rand);
    model.AddInput("nav_features", nav_features.data(), nav_features_size);

    std::array<float, nav_instructions_size> nav_instructions;
    std::generate(nav_instructions.begin(), nav_instructions.end(), rand);
    model.AddInput("nav_instructions", nav_instructions.data(), nav_instructions_size);

    std::array<float, features_buffer_size> features_buffer;
    std::generate(features_buffer.begin(), features_buffer.end(), rand);
    model.AddInput("features_buffer", features_buffer.data(), features_buffer_size);

    // std::vector<float> output;
    // model.AddOutput("outputs", output.data(), 6504*4);
    // for (auto out : output) {
    //     std::cout << out << std::endl;
    // }

    model.Run();
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

        // convert to YUV420
        cv::Mat yuv;
        cv::cvtColor(nframe, yuv, cv::COLOR_BGR2YUV_I420);
        // cv::cvtColor(yuv, yuv, cv::COLOR_YUV2BGR_NV12);
        // std::cout << yuv.total() << " " << yuv.size() << std::endl;
        // std::cout << "Resize rows: " << yuv.rows << " cols: " << yuv.cols << " Channel: " << yuv.channels() << std::endl;
    
        cv::imshow("Live YUV-NV12", yuv);

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