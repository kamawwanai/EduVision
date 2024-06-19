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

    // Получаем всех пользователей
    std::vector<User> users = userRepository.getAll();

    for (const User& user : users) {
        int label = user.getId();
        std::string user_data_path = "person_data/" + std::to_string(label) + "/";

        // Проходим по файлам в директории пользователя
        for (const auto& file : fs::directory_iterator(user_data_path)) {
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
    }

    std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
    serialize("models/face_descriptors.dat") << face_descriptors << labels;
}

void FaceRecognizer::addUserToModel(int userId) {
    std::vector<matrix<rgb_pixel>> new_faces;
    std::vector<int> new_labels;
    std::string user_data_path = "person_data/" + std::to_string(userId) + "/";

    // Проходим по файлам в директории пользователя
    for (const auto& file : fs::directory_iterator(user_data_path)) {
        if (file.path().extension() == ".jpg" || file.path().extension() == ".png") {
            matrix<rgb_pixel> img;
            load_image(img, file.path().string());

            std::vector<rectangle> dets = detector(img);
            if (dets.size() == 1) {
                auto shape = sp(img, dets[0]);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                new_faces.push_back(std::move(face_chip));
                new_labels.push_back(userId);
            }
        }
    }

    // Получаем текущие дескрипторы и метки из файла
    std::vector<matrix<float, 0, 1>> face_descriptors;
    std::vector<int> labels;
    try {
        dlib::deserialize("models/face_descriptors.dat") >> face_descriptors >> labels;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading face_descriptors.dat: " << e.what() << std::endl;
    }

    // Получаем дескрипторы для новых лиц
    std::vector<matrix<float, 0, 1>> new_face_descriptors = net(new_faces);

    // Добавляем новые данные к текущим
    face_descriptors.insert(face_descriptors.end(), new_face_descriptors.begin(), new_face_descriptors.end());
    labels.insert(labels.end(), new_labels.begin(), new_labels.end());

    // Сохраняем обновленные дескрипторы и метки в файл
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

void FaceRecognizer::recognizeFaces(cv::CascadeClassifier& face_cascade, std::vector<dlib::matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels, std::atomic<bool>& stop_flag) {
    while (!stop_flag) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_cond.wait(lock, [&] { return new_frame_ready || stop_flag; });

        if (stop_flag) break;

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

            int x1 = scaled_face.x;
            int y1 = scaled_face.y;

            if (label != -1) {
                if (userRepository.recognize(label)) {
                    auto user = *userRepository.findById(label);
                    std::string upd_user_info = user.getName() + " " + user.getSurname() + " " + user.getGroup() + " was recognized";
                    updated_users.push_back(upd_user_info);
                }
            }

            {
                /*std::lock_guard<std::mutex> lock(frame_mutex);
                cv::rectangle(current_frame, scaled_face, cv::Scalar(0, 255, 0), 2);
                cv::putText(current_frame, name, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                std::cout << name << std::endl;*/
            }
        }
    }
}

//CameraManager::CameraManager(FaceRecognizer& recognizer, cv::CascadeClassifier& face_cascade, std::vector<matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels, std::vector<std::string>& names)
//    : recognizer(recognizer), face_cascade(face_cascade), face_descriptors(face_descriptors), labels(labels), names(names) {
//}
//
//void CameraManager::start() {
//    cv::VideoCapture cap(0, cv::CAP_DSHOW);
//    if (!cap.isOpened()) {
//        throw std::runtime_error("Error opening video stream");
//    }
//
//    std::thread recognition_thread(&FaceRecognizer::recognizeFaces, &recognizer, std::ref(face_cascade), std::ref(face_descriptors), std::ref(labels), std::ref(names));
//
//    while (true) {
//        {
//            std::lock_guard<std::mutex> lock(recognizer.frame_mutex);
//            cap >> recognizer.current_frame;
//            if (recognizer.current_frame.empty()) {
//                break;
//            }
//        }
//
//        recognizer.new_frame_ready = true;
//        recognizer.frame_cond.notify_one();
//
//        {
//            std::lock_guard<std::mutex> lock(recognizer.frame_mutex);
//            cv::imshow("Face Recognition", recognizer.current_frame);
//        }
//
//        if (cv::waitKey(30) == 27) {
//            recognizer.stop = true;
//            recognizer.frame_cond.notify_one();
//            break;
//        }
//    }
//
//    recognition_thread.join();
//    cap.release();
//    cv::destroyAllWindows();
//}