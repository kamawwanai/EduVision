// EduVision.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include "FaceRecognition.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <GLFW/glfw3.h>

#include "User.hpp"

int main() {
    try {
        UserRepository userRepository;

        // Инициализация FaceRecognizer
        FaceRecognizer faceRecognizer(userRepository);

        // Загрузка модели каскадного классификатора для обнаружения лиц
        cv::CascadeClassifier face_cascade;
        if (!face_cascade.load("models/haarcascade_frontalface_default.xml")) {
            std::cerr << "Error loading haarcascade_frontalface_default.xml" << std::endl;
            return -1;
        }

        // Чтение обученных дескрипторов лиц и меток
        std::vector<matrix<float, 0, 1>> face_descriptors;
        std::vector<int> labels;
        std::vector<std::string> names{ "angeline_jolie", "brad_pitt", "denzel_washington", "hugh_jackman",
                                   "jennifer_lawrence", "johnny_depp", "kate_winslet", "leonardo_dicaprio",
                                   "magan_fox", "natalie_portman", "nicole_kidman", "robert_downey_jr",
                                   "sandra_bullock", "scarlett_johansson", "tom_cruise", "tom_hanks",
                                   "will_smith", "ksenia_karimova", "roma_kislitsyn", "lera_krasovskaya",
                                   "dima_kutuzov" };

        try {
            deserialize("models/face_descriptors.dat") >> face_descriptors >> labels;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading face descriptors: " << e.what() << std::endl;
            return -1;
        }

        // Инициализация CameraManager и запуск распознавания лиц
        CameraManager cameraManager(faceRecognizer, face_cascade, face_descriptors, labels, names);
        cameraManager.start();

        auto allUsers = userRepository.getAll();
        for (const auto& user : allUsers) {
            std::cout << user << std::endl;
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"
