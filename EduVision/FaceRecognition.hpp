#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/string.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "User.hpp"

namespace fs = std::filesystem;
using namespace dlib;


template <template <int, template<typename>class, int, typename> class block, int
    N, template<typename>class BN, typename SUBNET> using residual =
    add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int
    N, template<typename>class BN, typename SUBNET> using residual_down =
    add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares =
relu<residual<block, N, affine, SUBNET>>; template <int N, typename SUBNET> using
ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 =
ares<256, ares<256, ares_down<256, SUBNET>>>; template <typename SUBNET> using
alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>; template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>; template
<typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;


class FaceRecognizer {
public:
    FaceRecognizer(UserRepository& userRepository);
    void trainModel();
    void addUserToModel(int userId);
    void recognizeFaces(cv::CascadeClassifier& face_cascade, std::vector<matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels, std::atomic<bool>& stop_flag);

    static dlib::frontal_face_detector detector;
    std::mutex frame_mutex;
    cv::Mat current_frame;
    std::atomic<bool> new_frame_ready{ false };
    std::atomic<bool> stop{ false };
    std::condition_variable frame_cond;
    void markAttendance(int userId);

    std::vector<std::string> updated_users;

private:
    dlib::shape_predictor sp;
    anet_type net;
    UserRepository& userRepository;
};


//class CameraManager {
//public:
//    CameraManager(FaceRecognizer& recognizer, cv::CascadeClassifier& face_cascade, std::vector<matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels, std::vector<std::string>& names);
//    void start();
//
//private:
//    FaceRecognizer& recognizer;
//    cv::CascadeClassifier& face_cascade;
//    std::vector<matrix<float, 0, 1>>& face_descriptors;
//    std::vector<int>& labels;
//    std::vector<std::string>& names;
//};