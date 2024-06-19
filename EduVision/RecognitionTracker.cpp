#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#include "RecognitionTracker.hpp"
#include "User.hpp"

RecognitionTracker::RecognitionTracker(UserRepository& repository)
    : _repository(repository) {}

bool RecognitionTracker::recognize(int user_id) {
    auto now = std::chrono::system_clock::now();
    auto it = _last_recognition_time.find(user_id);

    // Получить последнее посещение из базы данных
    auto user_opt = _repository.findById(user_id);
    if (!user_opt) {
        return false; // Пользователь не найден
    }

    const auto& user = *user_opt;
    auto attendance = user.getAttendance();
    bool time_condition = false;

    if (!attendance.empty()) {
        auto last_attendance = std::chrono::system_clock::from_time_t(attendance.back());
        time_condition = (now - last_attendance) >= TIME_THRESHOLD;
    }
    else {
        time_condition = true; // Если нет записей о посещениях, допускаем новое посещение
    }

    if (it == _last_recognition_time.end() || time_condition) {
        _recognition_counts[user_id]++;
        _last_recognition_time[user_id] = now;
        bool mark = markAttendedIfThresholdReached(user_id);
        return mark;
    }
}

 bool RecognitionTracker::markAttendedIfThresholdReached(int user_id) {
    if (_recognition_counts[user_id] >= RECOGNITION_THRESHOLD) {
        if (auto user = _repository.findById(user_id)) {
            user->markAttended();
            _repository.update(*user);
            _recognition_counts[user_id] = 0; // Сбросить счетчик после отметки посещаемости

            return true;
        }
        return false;
    }
    return false;
}