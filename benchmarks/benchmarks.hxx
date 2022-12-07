#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>

/**
 * @brief Filter the arguments by removing the strings.
 *
 * @tparam t string typename.
 * @param argc Argument count.
 * @param argv Arguments.
 * @param s Variadic parameters.
 * @return auto A vector of char* with s removed.
 */
template <typename... t>
auto filtered_argv(int argc, char** argv, t&... s) {
  std::vector<std::string> args(argv, argv + argc);
  auto condition = [s...](const std::string& arg) -> bool {
    return ((arg == s) || ...);
  };
  args.erase(std::remove_if(args.begin(), args.end(), condition), args.end());

  std::vector<char*> cstrings;
  cstrings.reserve(args.size());

  for (auto& s : args)
    cstrings.push_back(&s[0]);

  return cstrings;
}

void print_arg(int argc, char** argv) {
  std::cout << "name of program: " << argv[0] << '\n';
  if (argc > 1) {
    std::cout << "there are " << argc - 1 << " (more) arguments, they are:\n";
    std::copy(argv + 1, argv + argc,
              std::ostream_iterator<const char*>(std::cout, "\n"));
  }
}