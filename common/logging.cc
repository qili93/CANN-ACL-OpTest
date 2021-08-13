#include "common/logging.h"

void gen_log(std::ostream& log_stream_, const char* file, const char* func, int lineno, const char* level, const int kMaxLen) {
  const int len = strlen(file);

  std::string time_str;
  struct tm tm_time;  // Time of creation of LogMessage
  time_t timestamp = time(NULL);
  localtime_r(&timestamp, &tm_time);

  // print date / time
  log_stream_ << '[' << level << ' ' << std::setw(2) << 1 + tm_time.tm_mon
              << '/' << std::setw(2) << tm_time.tm_mday << ' ' << std::setw(2)
              << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
              << std::setw(2) << tm_time.tm_sec << '.' << std::setw(3)
              << " ";

  if (len > kMaxLen) {
    log_stream_ << "..." << file + len - kMaxLen << ":" << lineno << " " << func
                << "] ";
  } else {
    log_stream_ << file << " " << func << ":" << lineno << "] ";
  }
}