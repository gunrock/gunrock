#include <string>

namespace gunrock {
namespace io {

#ifndef GIT_SHA1
#define GIT_SHA1 "unset"
#endif

std::string git_commit_sha1() {
  return GIT_SHA1;
}

}  // namespace io
}  // namespace gunrock