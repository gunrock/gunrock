// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_parameters.cu
 *
 * @brief Simple test driver program for util/parameters class.
 */

#include <gunrock/util/parameters.h>

using namespace gunrock;
using namespace gunrock::util;

int main(int argc, char* const argv[]) {
  Parameters parameters("Test program for util::Parameter class");
  parameters.Use("help", OPTIONAL_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER,
                 false, "Print this usage guide.", __FILE__, __LINE__);
  parameters.Use(
      "graph-type", REQUIRED_ARGUMENT | SINGLE_VALUE | REQUIRED_PARAMETER, "",
      "Type of graph to be processed, <market | rmat | rgg | smallworld>",
      &typeid(std::string), __FILE__, __LINE__);
  parameters.Use("market-file",
                 REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, "",
                 "path of the market graph file", &typeid(std::string),
                 __FILE__, __LINE__);
  parameters.Use("device", REQUIRED_ARGUMENT | MULTI_VALUE | OPTIONAL_PARAMETER,
                 0, "GPU device indices used for testing", __FILE__, __LINE__);

  // parameters.Use("ozbn", OPTIONAL_PARAMETER | ZERO_VALUE | NO_ARGUMENT,
  //    false, "Optional single bool value, no argument", __FILE__, __LINE__);
  parameters.Use("osbo", OPTIONAL_PARAMETER | SINGLE_VALUE | OPTIONAL_ARGUMENT,
                 true, "Optional single bool value, optional argument",
                 __FILE__, __LINE__);
  parameters.Use("osbr", OPTIONAL_PARAMETER | SINGLE_VALUE | REQUIRED_ARGUMENT,
                 true, "Optional single bool value, optional argument",
                 __FILE__, __LINE__);
  parameters.Use("ombo", OPTIONAL_PARAMETER | MULTI_VALUE | OPTIONAL_ARGUMENT,
                 false, "Optional multiple bool values, optional argument",
                 __FILE__, __LINE__);
  parameters.Use("ombr", OPTIONAL_PARAMETER | MULTI_VALUE | REQUIRED_ARGUMENT,
                 false, "Optional multiple bool values, required argument",
                 __FILE__, __LINE__);
  // parameters.Use("rzbn", REQUIRED_PARAMETER | ZERO_VALUE | NO_ARGUMENT,
  //    false, "Required single bool value, no argument", __FILE__, __LINE__);
  parameters.Use("rsbo", REQUIRED_PARAMETER | SINGLE_VALUE | OPTIONAL_ARGUMENT,
                 true, "Required single bool value, optional argument",
                 __FILE__, __LINE__);
  parameters.Use("rsbr", REQUIRED_PARAMETER | SINGLE_VALUE | REQUIRED_ARGUMENT,
                 true, "Required single bool value, optional argument",
                 __FILE__, __LINE__);
  parameters.Use("rmbo", REQUIRED_PARAMETER | MULTI_VALUE | OPTIONAL_ARGUMENT,
                 false, "Required multiple bool values, optional argument",
                 __FILE__, __LINE__);
  parameters.Use("rmbr", REQUIRED_PARAMETER | MULTI_VALUE | REQUIRED_ARGUMENT,
                 false, "Required multiple bool values, required argument",
                 __FILE__, __LINE__);

  // parameters.Use("ozin", OPTIONAL_PARAMETER | ZERO_VALUE | NO_ARGUMENT,
  //    1, "Optional single int value, no argument", __FILE__, __LINE__);
  parameters.Use("osio", OPTIONAL_PARAMETER | SINGLE_VALUE | OPTIONAL_ARGUMENT,
                 2, "Optional single int value, optional argument", __FILE__,
                 __LINE__);
  parameters.Use("osir", OPTIONAL_PARAMETER | SINGLE_VALUE | REQUIRED_ARGUMENT,
                 3, "Optional single int value, required argument", __FILE__,
                 __LINE__);
  parameters.Use("omio", OPTIONAL_PARAMETER | MULTI_VALUE | OPTIONAL_ARGUMENT,
                 4, "Optional multiple int values, optional argument", __FILE__,
                 __LINE__);
  parameters.Use("omir", OPTIONAL_PARAMETER | MULTI_VALUE | REQUIRED_ARGUMENT,
                 5, "Optional multiple int values, required argument", __FILE__,
                 __LINE__);
  // parameters.Use("rzin", REQUIRED_PARAMETER | ZERO_VALUE | NO_ARGUMENT,
  //    6, "Required single int value, no argument", __FILE__, __LINE__);
  parameters.Use("rsio", REQUIRED_PARAMETER | SINGLE_VALUE | OPTIONAL_ARGUMENT,
                 7, "Required single int value, optional argument", __FILE__,
                 __LINE__);
  parameters.Use("rsir", REQUIRED_PARAMETER | SINGLE_VALUE | REQUIRED_ARGUMENT,
                 8, "Required single int value, required argument", __FILE__,
                 __LINE__);
  parameters.Use("rmio", REQUIRED_PARAMETER | MULTI_VALUE | OPTIONAL_ARGUMENT,
                 9, "Required multiple int values, optional argument", __FILE__,
                 __LINE__);
  parameters.Use("rmir", REQUIRED_PARAMETER | MULTI_VALUE | REQUIRED_ARGUMENT,
                 10, "Required multiple int values, required argument",
                 __FILE__, __LINE__);

  parameters.Use("duplicated",
                 OPTIONAL_PARAMETER | SINGLE_VALUE | OPTIONAL_ARGUMENT, "null",
                 "Duplicated parameter defination", __FILE__, __LINE__);
  parameters.Use("duplicated",
                 OPTIONAL_PARAMETER | SINGLE_VALUE | OPTIONAL_ARGUMENT, "null",
                 "Duplicated parameter defination", __FILE__, __LINE__);

  parameters.Use("required",
                 REQUIRED_PARAMETER | SINGLE_VALUE | OPTIONAL_ARGUMENT, "",
                 "Required bool parameter", &typeid(bool), __FILE__, __LINE__);

  parameters.Print_Help();

  std::cout << std::endl;
  parameters.Parse_CommandLine(argc, argv);
  parameters.Check_Required();

  /*std::map<std::string, std::string> parameter_list = parameters.List();
  std::cout << std::endl;
  for (auto it = parameter_list.begin(); it != parameter_list.end(); it++)
  {
      std::cout << it -> first << " = " << it -> second << std::endl;
  }
  parameter_list.clear();*/

  /*parameters.Set("rsio", 10);
  std::cout << "parameters.Get<int>(\"rsio\") = " << parameters.Get<int>("rsio")
  << std::endl;*/

  /*std::vector<int> test_vec;
  test_vec.resize(10);
  test_vec[0] = 1; test_vec[1] = 2;
  test_vec[2] = 8; test_vec[3] = 10;
  std::cout << test_vec << std::endl;

  std::string test_vec_str = "9,8,7,6";
  std::istringstream istr(test_vec_str);
  istr >> test_vec;
  std::cout << test_vec << std::endl;*/

  parameters.Set("osio", "try");
  parameters.Get<bool>("rmir");

  std::cout << std::endl;
  std::cout << "Get<bool>(\"osbo\") = "
            << (parameters.Get<bool>("osbo") ? "true" : "false") << std::endl;
  std::cout << "Get<bool>(\"osbr\") = "
            << (parameters.Get<bool>("osbr") ? "true" : "false") << std::endl;
  std::cout << "Get<bool>(\"ombo\") = "
            << (parameters.Get<bool>("ombo") ? "true" : "false") << std::endl;
  std::cout << "Get<std::vector<bool> >(\"ombo\") = "
            << parameters.Get<std::vector<bool> >("ombo") << std::endl;
  std::cout << "Get<bool>(\"ombr\") = "
            << (parameters.Get<bool>("ombr") ? "true" : "false") << std::endl;
  std::cout << "Get<std::vector<bool> >(\"ombr\") = "
            << parameters.Get<std::vector<bool> >("ombr") << std::endl;
  std::cout << "Get<bool>(\"rsbo\") = "
            << (parameters.Get<bool>("rsbo") ? "true" : "false") << std::endl;
  std::cout << "Get<bool>(\"rsbr\") = "
            << (parameters.Get<bool>("rsbr") ? "true" : "false") << std::endl;
  std::cout << "Get<bool>(\"rmbo\") = "
            << (parameters.Get<bool>("rmbo") ? "true" : "false") << std::endl;
  std::cout << "Get<std::vector<bool> >(\"rmbo\") = "
            << parameters.Get<std::vector<bool> >("rmbo") << std::endl;
  std::cout << "Get<bool>(\"rmbr\") = "
            << (parameters.Get<bool>("rmbr") ? "true" : "false") << std::endl;
  std::cout << "Get<std::vector<bool> >(\"rmbr\") = "
            << parameters.Get<std::vector<bool> >("rmbr") << std::endl;

  std::cout << "Get<int>(\"osio\") = " << parameters.Get<int>("osio")
            << std::endl;
  std::cout << "Get<int>(\"osir\") = " << parameters.Get<int>("osir")
            << std::endl;
  std::cout << "Get<int>(\"omio\") = " << parameters.Get<int>("omio")
            << std::endl;
  std::cout << "Get<std::vector<int> >(\"omio\") = "
            << parameters.Get<std::vector<int> >("omio") << std::endl;
  std::cout << "Get<int>(\"omir\") = " << parameters.Get<int>("omir")
            << std::endl;
  std::cout << "Get<std::vector<int> >(\"omir\") = "
            << parameters.Get<std::vector<int> >("omir") << std::endl;
  std::cout << "Get<int>(\"rsio\") = " << parameters.Get<int>("rsio")
            << std::endl;
  std::cout << "Get<int>(\"rsir\") = " << parameters.Get<int>("rsir")
            << std::endl;
  std::cout << "Get<int>(\"rmio\") = " << parameters.Get<int>("rmio")
            << std::endl;
  std::cout << "Get<std::vector<int> >(\"rmio\") = "
            << parameters.Get<std::vector<int> >("rmio") << std::endl;
  std::cout << "Get<int>(\"rmir\") = " << parameters.Get<int>("rmir")
            << std::endl;
  std::cout << "Get<std::vector<int> >(\"rmir\") = "
            << parameters.Get<std::vector<int> >("rmir") << std::endl;
  return 0;
}
