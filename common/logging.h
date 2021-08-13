#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <iomanip>
#include <sstream>

#define LOG(status) LOG_##status.stream()
#define LOG_INFO LogMessage(__FILE__, __FUNCTION__, __LINE__, "I")
#define LOG_ERROR LOG_INFO
#define LOG_WARNING LogMessage(__FILE__, __FUNCTION__, __LINE__, "W")
#define LOG_FATAL LogMessageFatal(__FILE__, __FUNCTION__, __LINE__)

// VLOG()
#define VLOG(level) VLogMessage(__FILE__, __FUNCTION__, __LINE__, level).stream()

#define CHECK(x) if (!(x)) LogMessageFatal(__FILE__, __FUNCTION__, __LINE__).stream() << "Check failed: " #x << ": " // NOLINT(*)
#define _CHECK_BINARY(x, cmp, y) CHECK((x cmp y)) << (x) << "!" #cmp << (y) << " " // NOLINT(*)

#define CHECK_EQ(x, y) _CHECK_BINARY(x, ==, y)
#define CHECK_NE(x, y) _CHECK_BINARY(x, !=, y)
#define CHECK_LT(x, y) _CHECK_BINARY(x, <, y)
#define CHECK_LE(x, y) _CHECK_BINARY(x, <=, y)
#define CHECK_GT(x, y) _CHECK_BINARY(x, >, y)
#define CHECK_GE(x, y) _CHECK_BINARY(x, >=, y)

template <typename T>
static std::string to_string(const T& v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

void gen_log(std::ostream& log_stream_,
             const char* file,
             const char* func,
             int lineno,
             const char* level,
             const int kMaxLen = 40);

// LogMessage
class LogMessage {
 public:
  LogMessage(const char* file,
             const char* func,
             int lineno,
             const char* level = "I") {
    level_ = level;
    gen_log(log_stream_, file, func, lineno, level);
  }

  ~LogMessage() {
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  std::string level_;

  LogMessage(const LogMessage&) = delete;
  void operator=(const LogMessage&) = delete;
};

// LogMessageFatal
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file,
                  const char* func,
                  int lineno,
                  const char* level = "F")
      : LogMessage(file, func, lineno, level) {}

  ~LogMessageFatal() {
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }
};

class VLogMessage {
 public:
  VLogMessage(const char* file,
              const char* func,
              int lineno,
              const int32_t level_int = 0) {
    const char* GLOG_v = std::getenv("GLOG_v");
    GLOG_v_int = (GLOG_v && atoi(GLOG_v) > 0) ? atoi(GLOG_v) : 0;
    this->level_int = level_int;
    if (GLOG_v_int < level_int) {
      return;
    }
    const char* level = to_string(level_int).c_str();
    gen_log(log_stream_, file, func, lineno, level);
  }

  ~VLogMessage() {
    if (GLOG_v_int < this->level_int) {
      return;
    }
    log_stream_ << '\n';
    fprintf(stderr, "%s", log_stream_.str().c_str());
  }

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  int32_t GLOG_v_int;
  int32_t level_int;

  VLogMessage(const VLogMessage&) = delete;
  void operator=(const VLogMessage&) = delete;
};