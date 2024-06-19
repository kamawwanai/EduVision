#pragma once

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

#include <sqlite_orm/sqlite_orm.h>
#include <ctime>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <chrono>
#include "RecognitionTracker.hpp"

class User {
    friend class UserRepository;

public:
    // Only for repository usage
    struct UserPersist {
        int _id;
        std::string _name;
        std::string _surname;
        std::string _patronymic;
        std::unique_ptr<std::string> _group;
        std::unique_ptr<std::string> _photo_path;
    };

    struct AttendancePersist {
        int _id = -1;
        int _user_id = -1;
        std::string _datetime;

        AttendancePersist() = default;

        explicit AttendancePersist(int user_id)
            : _user_id(user_id),
            _datetime(std::to_string(std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::now()))) {}
    };

public:
    // Only for repository usage
    explicit User(UserPersist&& userPersist) noexcept;

public:
    User(std::string name, std::string surname, std::string patronymic,
        std::string group, std::string photo_path);

    [[nodiscard]] auto getId() const -> int;

    [[nodiscard]] auto getName() const->std::string;

    [[nodiscard]] auto getSurname() const->std::string;

    [[nodiscard]] auto getPatronymic() const->std::string;

    [[nodiscard]] auto getGroup() const->std::string;

    void setGroup(std::string group);

    [[nodiscard]] auto getPhotoPath() const->std::string;

    auto markAttended() -> void;

    [[nodiscard]] auto getAttendance() const->std::vector<std::time_t>;

private:
    UserPersist _persist;
    std::vector<AttendancePersist> _attendance;
};

auto operator<<(std::ostream& os, const User& c)->std::ostream&;

class UserRepository {
public:
    UserRepository();

    void create(User& user);

    void update(User& user);

    void remove(int id);

    void clearDatabase();

    bool recognize(int user_id);

    [[nodiscard]] auto findById(int id) const->std::optional<User>;

    [[nodiscard]] auto getAll() const->std::vector<User>;

    [[nodiscard]] auto getAllByGroup(std::string group) const->std::vector<User>;

    [[nodiscard]] auto findUserByFullName(std::string name, std::string surname) const->std::optional<User>;

private:
    void enrich_attendance(int id, User& user) const;

    RecognitionTracker _recognitionTracker;
};