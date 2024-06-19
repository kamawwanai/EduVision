#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#include "User.hpp"
#include <sqlite_orm/sqlite_orm.h>
#include <memory>

User::User(std::string name, std::string surname, std::string patronymic,
    std::string group, std::string photo_path)
    : _persist{ 0,
               std::move(name),
               std::move(surname),
               std::move(patronymic),
               std::make_unique<std::string>(std::move(group)),
               std::make_unique<std::string>(std::move(photo_path)) } {}

User::User(UserPersist&& userPersist) noexcept
    : _persist(std::move(userPersist)) {}

auto User::getId() const -> int { return _persist._id; }

auto User::getName() const -> std::string { return _persist._name; }

auto User::getSurname() const -> std::string { return _persist._surname; }

auto User::getPatronymic() const -> std::string { return _persist._patronymic; }

auto User::getGroup() const -> std::string { return { *_persist._group }; }

void User::setGroup(std::string group) {
    _persist._group = std::make_unique<std::string>(std::move(group));
}

auto User::getPhotoPath() const -> std::string {
    return { *_persist._photo_path };
}

auto User::markAttended() -> void { _attendance.emplace_back(_persist._id); }

auto User::getAttendance() const -> std::vector<std::time_t> {
    std::vector<std::time_t> out;
    out.reserve(_attendance.size());
    for (const auto& attendance : _attendance) {
        out.push_back(std::stoll(attendance._datetime));
    }
    return out;
}

auto operator<<(std::ostream& os, const User& c) -> std::ostream& {
    auto& out = os << "User { id: " << c.getId() << ", name: " << c.getName()
        << ", surname: " << c.getSurname()
        << ", patronymic: " << c.getPatronymic()
        << ", group: " << c.getGroup()
        << ", photo_path: " << c.getPhotoPath() << " }"
        << ", attendance: [";
    for (time_t attendance : c.getAttendance()) {
        auto time_formatted = std::string{ std::ctime(&attendance) }; //NOLINT
        time_formatted.pop_back();
        out << time_formatted << ", ";
    }
    return out << "]";
}

static auto storage = sqlite_orm::make_storage(
    "local.db",
    sqlite_orm::make_table(
        "users",
        sqlite_orm::make_column("id", &User::UserPersist::_id,
            sqlite_orm::primary_key().autoincrement()),
        sqlite_orm::make_column("name", &User::UserPersist::_name),
        sqlite_orm::make_column("surname", &User::UserPersist::_surname),
        sqlite_orm::make_column("patronymic", &User::UserPersist::_patronymic),
        sqlite_orm::make_column("group", &User::UserPersist::_group),
        sqlite_orm::make_column("photo_path", &User::UserPersist::_photo_path)),
    sqlite_orm::make_table(
        "attendance",
        sqlite_orm::make_column("id", &User::AttendancePersist::_id,
            sqlite_orm::primary_key().autoincrement()),
        sqlite_orm::make_column("user_id", &User::AttendancePersist::_user_id),
        sqlite_orm::make_column("datetime",
            &User::AttendancePersist::_datetime)));

void UserRepository::create(User& user) {
    int id = storage.insert(user._persist);
    user._persist._id = id;
    for (auto& attendance : user._attendance) {
        attendance._user_id = id;
        int attendance_id = storage.insert(attendance);
        attendance._id = attendance_id;
    }
}

void UserRepository::update(User& user) {
    storage.update(user._persist);
    for (auto& attendance : user._attendance) {
        if (attendance._id == -1) {
            int attendance_id = storage.insert(attendance);
            attendance._id = attendance_id;
        }
    }
}

void UserRepository::remove(int id) {
    storage.remove<User::UserPersist>(id);
    storage.remove_all<User::AttendancePersist>(sqlite_orm::where(
        sqlite_orm::c(&User::AttendancePersist::_user_id) == id));
}

void UserRepository::enrich_attendance(int id, User& user) const {
    using namespace sqlite_orm; // NOLINT 
    auto attendance_persists = storage.get_all<User::AttendancePersist>(
        where(c(&User::AttendancePersist::_user_id) == id));
    for (auto& attendancePersist : attendance_persists) {
        user._attendance.emplace_back(std::move(attendancePersist));
    }
}

auto UserRepository::findById(int id) const -> std::optional<User> {
    try {
        auto user_persist = storage.get<User::UserPersist>(id);
        auto user = User{ std::move(user_persist) };
        enrich_attendance(id, user);
        return user;
    }
    catch (std::system_error& e) {
        auto error_code = e.code();
        if (error_code == sqlite_orm::orm_error_code::not_found) {
            return std::nullopt;
        }
        throw;
    }
}

auto UserRepository::getAll() const -> std::vector<User> {
    std::vector<User> users;
    for (auto& userPersist : storage.get_all<User::UserPersist>()) {
        auto user = User(std::move(userPersist));
        enrich_attendance(user.getId(), user);
        users.emplace_back(std::move(user));
    }
    return users;
}

auto UserRepository::getAllByGroup(std::string group) const -> std::vector<User> {
    using namespace sqlite_orm; // NOLINT
    std::vector<User> users;
    for (auto& userPersist : storage.get_all<User::UserPersist>(where(c(&User::UserPersist::_group) == group))) {
        auto user = User(std::move(userPersist));
        enrich_attendance(user.getId(), user);
        users.emplace_back(std::move(user));
    }
    return users;
}

auto UserRepository::findUserByFullName(std::string name, std::string surname) const -> std::optional<User> {
    using namespace sqlite_orm; // NOLINT
    try {
        auto user_persist = storage.get<User::UserPersist>(
            where(c(&User::UserPersist::_name) == name and c(&User::UserPersist::_surname) == surname));
        auto user = User{ std::move(user_persist) };
        enrich_attendance(user.getId(), user);
        return user;
    }
    catch (std::system_error& e) {
        auto error_code = e.code();
        if (error_code == sqlite_orm::orm_error_code::not_found) {
            return std::nullopt;
        }
        throw;
    }
}

void UserRepository::clearDatabase() {
    storage.remove_all<User::AttendancePersist>();
    storage.remove_all<User::UserPersist>();
}

UserRepository::UserRepository() : _recognitionTracker(*this) { storage.sync_schema(); }

bool UserRepository::recognize(int user_id) {
    return _recognitionTracker.recognize(user_id);
}