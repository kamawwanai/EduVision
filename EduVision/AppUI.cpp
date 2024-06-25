#include "AppUI.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <atomic>


#if defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>
#endif


AppUI::AppUI(UserRepository& dataBase, FaceRecognizer& recognizer, cv::CascadeClassifier& face_cascade, std::vector<matrix<float, 0, 1>>& face_descriptors, std::vector<int>& labels)
    : dataBase(dataBase), recognizer(recognizer), face_cascade(face_cascade), face_descriptors(face_descriptors), labels(labels) {
}

void AppUI::start() {
    // Initialize OpenCV video capture
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error opening video stream");
    }

    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    const char* glsl_version = "#version 130";
    GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGui OpenCV Example", NULL, NULL);
    if (window == NULL) {
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader (GLAD in this case)
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize OpenGL loader!");
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImFont* font_default = io.Fonts->AddFontDefault();
    ImFont* font_huge = io.Fonts->AddFontFromFileTTF("fonts/Helvetica-Bold.ttf", 48.0f);
    ImFont* font_large = io.Fonts->AddFontFromFileTTF("fonts/Helvetica-Bold.ttf", 36.0f);
    ImFont* font_medium = io.Fonts->AddFontFromFileTTF("fonts/Helvetica.ttf", 24.0f);
    ImFont* font_small = io.Fonts->AddFontFromFileTTF("fonts/Helvetica.ttf", 16.0f);

    // Start face recognition thread
    std::atomic<bool> stop_flag{ false };
    std::thread recognition_thread(&FaceRecognizer::recognizeFaces, &recognizer, std::ref(face_cascade), std::ref(face_descriptors), std::ref(labels), std::ref(stop_flag));

    bool show_group_attendance_popup = false;
    bool group_attendance_error = false;
    std::vector<User> group_users;

    bool show_student_attendance_popup = false;
    bool student_attendance_error = false;
    std::optional<User> selected_student;

    char group_attendance[96] = "";
    char student_attendance[96] = "";

    bool show_add_student_popup = false;

    char new_surname[96] = "";
    char new_name[96] = "";
    char new_patronymic[96] = "";
    char new_group[96] = "";
    std::string new_user_photo_path;
    bool user_created = false;
    int new_user_id = -1;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        {
            std::lock_guard<std::mutex> lock(recognizer.frame_mutex);
            cap >> recognizer.current_frame;
            if (recognizer.current_frame.empty()) {
                stop_flag = true;
                break;
            }
        }

        recognizer.new_frame_ready = true;
        recognizer.frame_cond.notify_one();

        std::vector<std::string> recognized_users = recognizer.updated_users;

        {
            std::lock_guard<std::mutex> lock(recognizer.frame_mutex);

            // Convert the frame to RGBA
            cv::Mat frame_rgba;
            cv::cvtColor(recognizer.current_frame, frame_rgba, cv::COLOR_BGR2RGBA);

            // Create a texture from the frame
            GLuint texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame_rgba.cols, frame_rgba.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_rgba.data);

            // Start the ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus;
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);

            ImGui::SetNextWindowSize(ImVec2((float)display_w, (float)display_h));
            ImGui::SetNextWindowPos(ImVec2(0, 0));

            // Create a window called "Face Recognition"
            ImGui::Begin("Face Recognition", nullptr, window_flags);

            ImGui::PushFont(font_huge);
            ImGui::Text("EduVision");
            ImGui::PopFont();

            ImGui::Dummy(ImVec2(0.0f, 12.0f));

            // Display the frame as an image
            ImGui::Image((void*)(intptr_t)texture, ImVec2((float)frame_rgba.cols, (float)frame_rgba.rows));

            ImGui::Dummy(ImVec2(0.0f, 16.0f));

            ImGui::PushFont(font_medium);

            for (auto it = recognized_users.rbegin(); it != recognized_users.rend(); ++it) {
                ImGui::Text("%s", it->c_str());
            }
            ImGui::PopFont();

            ImGui::End();

            ImGuiWindowFlags controls_window_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground;

            ImGui::SetNextWindowSize(ImVec2((float)(display_w - frame_rgba.cols - 40), (float)frame_rgba.rows));
            ImGui::SetNextWindowPos(ImVec2((float)frame_rgba.cols + 40, 0));

            // Create a new window for input fields and buttons
            ImGui::Begin("Controls", nullptr, controls_window_flags);

            ImGui::Dummy(ImVec2(0.0f, 60.0f));

            ImGui::PushFont(font_large);
            ImGui::Text("Information");
            ImGui::PopFont();

            ImGui::Dummy(ImVec2(0.0f, 16.0f));

            ImGui::PushFont(font_medium);

            // Group attendance input field and button
            ImGui::Text("Group attendance");
            ImGui::SameLine();
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));
            ImGui::InputText("##group_attendance", group_attendance, IM_ARRAYSIZE(group_attendance));
            ImGui::PopStyleVar();
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(16, 8));
            if (ImGui::Button("Get##group")) {
                group_users = dataBase.getAllByGroup(group_attendance);
                if (group_users.empty()) {
                    group_attendance_error = true;
                }
                else {
                    group_attendance_error = false;
                }
                show_group_attendance_popup = true;
                ImGui::OpenPopup("Group Attendance");
            }
            ImGui::PopStyleVar();

            if (show_group_attendance_popup) {
                ImGui::SetNextWindowSize(ImVec2(800, 600));
                if (ImGui::BeginPopupModal("Group Attendance", &show_group_attendance_popup, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_AlwaysHorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
                    if (group_attendance_error) {
                        ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(255, 0, 0, 255));
                        ImGui::Text("Wrong group name");
                        ImGui::PopStyleColor();
                    }
                    else {
                        std::map<std::time_t, int> date_map;
                        for (const auto& user : group_users) {
                            for (const auto& attendance : user.getAttendance()) {
                                std::tm* tm_ptr = std::localtime(&attendance);
                                tm_ptr->tm_sec = 0;
                                tm_ptr->tm_min = 0;
                                tm_ptr->tm_hour = 0;
                                std::time_t date = std::mktime(tm_ptr);
                                date_map[date]++;
                            }
                        }

                        std::vector<std::time_t> sorted_dates;
                        for (const auto& entry : date_map) {
                            sorted_dates.push_back(entry.first);
                        }

                        if (ImGui::BeginTable("GroupAttendanceTable", 1 + sorted_dates.size(), ImGuiTableFlags_ScrollY)) {
                            ImGui::TableSetupColumn("Name");

                            for (const auto& date : sorted_dates) {
                                std::tm* tm_ptr = std::localtime(&date);
                                char date_str[20];
                                std::strftime(date_str, sizeof(date_str), "%Y-%m-%d", tm_ptr);
                                ImGui::TableSetupColumn(date_str);
                            }
                            ImGui::TableHeadersRow();

                            for (const auto& user : group_users) {
                                ImGui::TableNextRow();
                                ImGui::TableSetColumnIndex(0);
                                std::string full_name = user.getSurname() + " " + user.getName() + " " + user.getPatronymic();
                                ImGui::Text("%s", full_name.c_str());

                                std::map<std::time_t, int> user_attendance_map;
                                for (const auto& attendance : user.getAttendance()) {
                                    std::tm* tm_ptr = std::localtime(&attendance);
                                    tm_ptr->tm_sec = 0;
                                    tm_ptr->tm_min = 0;
                                    tm_ptr->tm_hour = 0;
                                    std::time_t date = std::mktime(tm_ptr);
                                    user_attendance_map[date]++;
                                }

                                for (size_t i = 0; i < sorted_dates.size(); ++i) {
                                    ImGui::TableSetColumnIndex(i + 1);
                                    int count = user_attendance_map[sorted_dates[i]];
                                    ImGui::Text("%d", count);
                                }
                            }

                            ImGui::EndTable();
                        }
                    }
                    if (ImGui::Button("Close")) {
                        ImGui::CloseCurrentPopup();
                        show_group_attendance_popup = false;
                    }
                    ImGui::EndPopup();
                }
            }

            ImGui::Dummy(ImVec2(0.0f, 20.0f));

            // Student attendance input field and button
            ImGui::Text("Student attendance");
            ImGui::SameLine();
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));
            ImGui::InputText("##student_attendance", student_attendance, IM_ARRAYSIZE(student_attendance));
            ImGui::PopStyleVar();
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(16, 8));
            if (ImGui::Button("Get##student")) {
                std::string full_name = student_attendance;
                std::string name, surname;
                size_t pos = full_name.find(' ');
                if (pos != std::string::npos) {
                    name = full_name.substr(0, pos);
                    surname = full_name.substr(pos + 1);
                }
                else {
                    student_attendance_error = true;
                    show_student_attendance_popup = true;
                    ImGui::OpenPopup("Student Attendance");
                    return;
                }
                selected_student = dataBase.findUserByFullName(name, surname);
                std::cout << (*selected_student).getId() << std::endl;
                if (!selected_student) {
                    student_attendance_error = true;
                }
                else {
                    student_attendance_error = false;
                }
                show_student_attendance_popup = true;
                ImGui::OpenPopup("Student Attendance");
            }

            if (show_student_attendance_popup) {
                ImGui::SetNextWindowSize(ImVec2(800, 600));
                if (ImGui::BeginPopupModal("Student Attendance", &show_student_attendance_popup, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
                    if (student_attendance_error) {
                        ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(255, 0, 0, 255));
                        ImGui::Text("Wrong user name");
                        ImGui::PopStyleColor();
                    }
                    else if (selected_student) {
                        const User& user = *selected_student;

                        // Собираем уникальные даты посещений
                        std::map<std::time_t, int> date_map;
                        for (const auto& attendance : user.getAttendance()) {
                            std::tm* tm_ptr = std::localtime(&attendance);
                            tm_ptr->tm_sec = 0;
                            tm_ptr->tm_min = 0;
                            tm_ptr->tm_hour = 0;
                            std::time_t date = std::mktime(tm_ptr);
                            date_map[date]++;
                        }

                        // Преобразуем map в отсортированный вектор дат
                        std::vector<std::time_t> sorted_dates;
                        for (const auto& entry : date_map) {
                            sorted_dates.push_back(entry.first);
                        }

                        // Отображаем таблицу
                        if (ImGui::BeginTable("StudentAttendanceTable", 1 + sorted_dates.size(), ImGuiTableFlags_ScrollY)) {
                            ImGui::TableSetupColumn("Date");

                            for (const auto& date : sorted_dates) {
                                std::tm* tm_ptr = std::localtime(&date);
                                char date_str[20];
                                std::strftime(date_str, sizeof(date_str), "%Y-%m-%d", tm_ptr);
                                ImGui::TableSetupColumn(date_str);
                            }
                            ImGui::TableHeadersRow();

                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            std::string full_name = user.getSurname() + " " + user.getName() + " " + user.getPatronymic();
                            ImGui::Text("%s", full_name.c_str());

                            for (size_t i = 0; i < sorted_dates.size(); ++i) {
                                ImGui::TableSetColumnIndex(i + 1);
                                int count = date_map[sorted_dates[i]];
                                ImGui::Text("%d", count);
                            }

                            ImGui::EndTable();
                        }
                    }
                    if (ImGui::Button("Close")) {
                        ImGui::CloseCurrentPopup();
                        show_student_attendance_popup = false;
                    }
                    ImGui::EndPopup();
                }
            }

            ImGui::PopStyleVar();

            ImGui::Dummy(ImVec2(0.0f, 32.0f));

            ImGui::PushFont(font_large);
            ImGui::Text("Tools");
            ImGui::PopFont();

            ImGui::Dummy(ImVec2(0.0f, 16.0f));

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(16, 8));
            // Add new student button
            if (ImGui::Button("Add new student")) {
                stop_flag = true;
                recognizer.frame_cond.notify_one();
                if (recognition_thread.joinable()) {
                    recognition_thread.join();
                }
                show_add_student_popup = true;
                ImGui::OpenPopup("Add New Student");
            }

            if (show_add_student_popup) {
                ImGui::SetNextWindowSize(ImVec2(800, 600));
                if (ImGui::BeginPopupModal("Add New Student", &show_add_student_popup, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
                    ImGui::InputText("Surname", new_surname, IM_ARRAYSIZE(new_surname));
                    ImGui::InputText("Name", new_name, IM_ARRAYSIZE(new_name));
                    ImGui::InputText("Patronymic", new_patronymic, IM_ARRAYSIZE(new_patronymic));
                    ImGui::InputText("Group", new_group, IM_ARRAYSIZE(new_group));
                    if (ImGui::Button("Create User")) {
                        try {
                            User new_user(new_name, new_surname, new_patronymic, new_group, "");
                            new_user_id = new_user.getId();
                            dataBase.create(new_user);
                            user_created = true;
                        }
                        catch (const std::exception& e) {
                            ImGui::Text("Error creating user: %s", e.what());
                            user_created = false;
                        }
                    }
                    if (user_created) {
                        ImGui::Text("User created successfully. ID: %d", new_user_id);
                        ImGui::Text("Please upload user's photos. Path:person_data/%d/", new_user_id);
                        if (ImGui::Button("I've uploaded photos")) {
                            if (ImGui::Button("Add user in Recognizer")) {
                                try {
                                    recognizer.addUserToModel(new_user_id);
                                    show_add_student_popup = false;
                                }
                                catch (const std::exception& e) {
                                    ImGui::Text("Error adding user to recognizer: %s", e.what());
                                }
                            }
                        }
                    }
                    if (ImGui::Button("Close")) {
                        show_add_student_popup = false;
                        stop_flag = false;
                        recognition_thread = std::thread(&FaceRecognizer::recognizeFaces, &recognizer, std::ref(face_cascade), std::ref(face_descriptors), std::ref(labels), std::ref(stop_flag));
                    }
                    ImGui::EndPopup();
                }
            }
            ImGui::PopStyleVar();

            ImGui::PopFont();

            ImGui::End();

            // Rendering
            ImGui::Render();
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.00f, 0.00f, 0.00f, 1.00f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);

            // Cleanup texture
            glDeleteTextures(1, &texture);
        }

        if (cv::waitKey(30) == 27) {
            stop_flag = true;
            recognizer.frame_cond.notify_one();
            break;
        }
    }

    stop_flag = true;
    recognizer.frame_cond.notify_one();
    if (recognition_thread.joinable()) {
        recognition_thread.join();
    }
    cap.release();

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
}