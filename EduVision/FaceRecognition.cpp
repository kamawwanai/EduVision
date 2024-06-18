#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#include "FaceRecognition.hpp"
#include "User.hpp"


dlib::frontal_face_detector FaceRecognizer::detector = dlib::get_frontal_face_detector();

FaceRecognizer::FaceRecognizer(UserRepository& userRepository) : userRepository(userRepository) {
    try {
        dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading shape_predictor_68_face_landmarks.dat: " << e.what() << std::endl;
        throw;
    }

    try {
        dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading dlib_face_recognition_resnet_model_v1.dat: " << e.what() << std::endl;
        throw;
    }
}

void FaceRecognizer::trainModel() {
    std::vector<matrix<rgb_pixel>> faces;
    std::vector<int> labels;

    std::string data_path = "person_data/";

    for (const auto& entry : fs::directory_iterator(data_path)) {
        if (entry.is_directory()) {
            int label = stoi(entry.path().filename().string());
            std::cout << "Label: " << label << " for directory: " << entry.path().string() << std::endl;
            for (const auto& file : fs::directory_iterator(entry.path())) {
                if (file.path().extension() == ".jpg" || file.path().extension() == ".png") {
                    matrix<rgb_pixel> img;
                    load_image(img, file.path().string());

                    std::vector<rectangle> dets = detector(img);
                    if (dets.size() == 1) {
                        auto shape = sp(img, dets[0]);
                        matrix<rgb_pixel> face_chip;
                        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                        faces.push_back(std::move(face_chip));
                        labels.push_back(label);
                    }
                }
            }
            label++;
        }
    }

    std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
    serialize("models/face_descriptors.dat") << face_descriptors << labels;
}

void FaceRecognizer::markAttendance(int userId) {
    auto user = userRepository.findById(userId);
    if (user) {
        auto attendance = user->getAttendance();
        if (!attendance.empty()) {
            auto last_attendance = attendance.back();
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            if (now - last_attendance < 1800) {
                return;  // Если последнее посещение было менее 30 минут назад, не отмечаем
            }
        }
        user->markAttended();
        userRepository.update(*user);
    }
}

void FaceRecognizer::recognizeFaces(cv::CascadeClassifier& face_cascade, std::vector<dlib::matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels, std::vector<std::string>& names) {
    while (!stop) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_cond.wait(lock, [&] { return new_frame_ready || stop; });

        if (stop) break;

        cv::Mat frame = current_frame.clone();
        new_frame_ready = false;
        lock.unlock();

        cv::Mat small_frame;
        cv::resize(frame, small_frame, cv::Size(frame.cols / 2, frame.rows / 2)); // Reduce image size for faster processing

        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cv::cvtColor(small_frame, gray, cv::COLOR_BGR2GRAY);
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

        for (auto& face : faces) {
            cv::Rect scaled_face(face.x * 2, face.y * 2, face.width * 2, face.height * 2); // Scale back to original size
            cv::Mat face_roi = frame(scaled_face);
            dlib::cv_image<dlib::bgr_pixel> cimg(face_roi);
            auto shape = sp(cimg, dlib::rectangle(0, 0, face_roi.cols, face_roi.rows));
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(cimg, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
            dlib::matrix<float, 0, 1> face_descriptor = net(face_chip);

            float min_distance = 0.6;
            int label = -1;
            for (size_t j = 0; j < face_descriptors.size(); ++j) {
                float distance = dlib::length(face_descriptor - face_descriptors[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    label = labels[j];
                }
            }

            std::string name = (label == -1) ? "Unknown" : names[label];
            int x1 = scaled_face.x;
            int y1 = scaled_face.y;

            if (label != -1) {
                userRepository.recognize(label);
            }

            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                cv::rectangle(current_frame, scaled_face, cv::Scalar(0, 255, 0), 2);
                cv::putText(current_frame, name, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                std::cout << name << std::endl;
            }
        }
    }
}

CameraManager::CameraManager(FaceRecognizer& recognizer, cv::CascadeClassifier& face_cascade, std::vector<matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels, std::vector<std::string>& names)
    : recognizer(recognizer), face_cascade(face_cascade), face_descriptors(face_descriptors), labels(labels), names(names) {
}

void CameraManager::start() {
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error opening video stream");
    }

    std::thread recognition_thread(&FaceRecognizer::recognizeFaces, &recognizer, std::ref(face_cascade), std::ref(face_descriptors), std::ref(labels), std::ref(names));

    while (true) {
        {
            std::lock_guard<std::mutex> lock(recognizer.frame_mutex);
            cap >> recognizer.current_frame;
            if (recognizer.current_frame.empty()) {
                break;
            }
        }

        recognizer.new_frame_ready = true;
        recognizer.frame_cond.notify_one();

        {
            std::lock_guard<std::mutex> lock(recognizer.frame_mutex);
            cv::imshow("Face Recognition", recognizer.current_frame);
        }

        if (cv::waitKey(30) == 27) {
            recognizer.stop = true;
            recognizer.frame_cond.notify_one();
            break;
        }
    }

    recognition_thread.join();
    cap.release();
    cv::destroyAllWindows();
}