#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#pragma once

#include <unordered_map>
#include <chrono>

class UserRepository;

class RecognitionTracker {
public:
    explicit RecognitionTracker(UserRepository& repository);

    void recognize(int user_id);

private:
    UserRepository& _repository;
    std::unordered_map<int, int> _recognition_counts;
    std::unordered_map<int, std::chrono::system_clock::time_point> _last_recognition_time;
    static constexpr int RECOGNITION_THRESHOLD = 5;
    static constexpr std::chrono::minutes TIME_THRESHOLD = std::chrono::minutes(30);

    void markAttendedIfThresholdReached(int user_id);
};