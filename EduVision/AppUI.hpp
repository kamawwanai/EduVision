#pragma once

#include "FaceRecognition.hpp"

class AppUI {
public:
    AppUI(UserRepository& dataBase, FaceRecognizer& recognizer, cv::CascadeClassifier& face_cascade, std::vector<matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels);
    void start();

private:
    UserRepository& dataBase;
    FaceRecognizer& recognizer;
    cv::CascadeClassifier& face_cascade;
    std::vector<matrix<float, 0, 1>>& face_descriptors;
    std::vector<int>& labels;
};