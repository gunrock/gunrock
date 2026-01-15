#include <gunrock/compat/runtime_api.h>
#include <gunrock/error.hxx>  // error checking
#include <gunrock/framework/problem.hxx>

#include <gtest/gtest.h>

using namespace gunrock;

TEST(framework, problem) {
  // Use fully qualified type to avoid MSVC parsing issues
  gunrock::error::error_t status = hipSuccess;
  // XXX ... write a test for [problem.hxx]
  (void)status;  // Suppress unused variable warning
}