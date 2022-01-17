#include <cxxopts.hpp>
#include <gunrock/util/filepath.hxx>

using namespace gunrock;

void test_cmd(int argc, char** argv) {
  cxxopts::Options options(argv[0], "Gunrock commandline parser test");
  options.add_options()  // Allows to add options.
      ("c,csr", "CSR binary file",
       cxxopts::value<std::string>())  // CSR
      ("m,market", "Matrix-market format file",
       cxxopts::value<std::string>())  // Market
      ("d,device", "Device to run on",
       cxxopts::value<int>()->default_value("0"))  // Device
      ("v,verbose", "Verbose output",
       cxxopts::value<bool>()->default_value("false"))  // Verbose (not used)
      ("h,help", "Print help");                         // Help

  auto result = options.parse(argc, argv);

  if (result.count("help") ||
      (result.count("market") == 0 && result.count("csr") == 0)) {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
  }

  if (result.count("market") == 1) {
    std::string filename = result["market"].as<std::string>();
    if (util::is_market(filename)) {
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }
  } else if (result.count("csr") == 1) {
    std::string filename = result["csr"].as<std::string>();
    if (util::is_binary_csr(filename)) {
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }
  } else {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
  }
}

int main(int argc, char** argv) {
  test_cmd(argc, argv);
}