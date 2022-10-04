namespace gunrock {
namespace util {
namespace stats {

#ifdef __clang__
std::string compiler = "Clang";
std::string compiler_version = std::to_string(
    (__clang_major__ % 100) * 10000000 + (__clang_minor__ % 100) * 100000 +
    (__clang_patchlevel__ % 100000));
#else
std::string compiler = "Gnu GCC C/C++";
#ifdef __GNUC_PATCHLEVEL__
std::string compiler_version = std::to_string((__GNUC__ % 100) * 10000000 +
                                              (__GNUC_MINOR__ % 100) * 100000 +
                                              (__GNUC_PATCHLEVEL__ % 100000));
#else
std::string compiler_version = std::to_string((__GNUC__ % 100) * 10000000 +
                                              (__GNUC_MINOR__ % 100) * 100000);
#endif
#endif

}  // namespace stats
}  // namespace util
}  // namespace gunrock