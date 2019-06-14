#include <iostream>
#include <algorithm>
#include <gunrock/util/binary_search.cuh>

cudaError_t BinarySearchTest() {
  cudaError_t retval = cudaSuccess;

  int num_elements = 2000;
  int min_element = 0;
  int max_element = 512;
  int num_tests = 1000;
  int element_range = max_element - min_element;

  int* elements = new int[num_elements];

  int num_errors = 0;

  srand(time(NULL));
  for (int i = 0; i < num_elements; i++)
    elements[i] = (rand() % element_range) + min_element;
  std::sort(elements, elements + num_elements);

  std::cout << "Testing BinarySearch_LeftMost" << std::endl;
  for (int i = 0; i < num_tests; i++) {
    int element = (rand() % (element_range + 2)) + min_element - 1;
    // std::cout << ".";
    int pos = gunrock::util::BinarySearch_LeftMost(element, elements, 0,
                                                   num_elements - 1);

    // if (element < min_element ||
    //    element >= min_element + element_range ||
    //    pos < 0 ||
    //    pos >= num_elements)
    //    std::cout << "Edge " << element << ": pos = " << pos << " ..."
    //        << (pos <= 0 ? -1 : elements[pos - 1]) << " , "
    //        << (pos < 0 || pos >= num_elements ? -1 : elements[pos]) << " , "
    //        << (pos + 1 >= num_elements ? -1 : elements[pos + 1]) <<
    //        std::endl;

    if ((pos - 1 >= 0 && pos - 1 < num_elements &&
         element <= elements[pos - 1]) ||
        (pos >= 0 && pos < num_elements && element < elements[pos]) ||
        (pos >= 0 && pos + 1 < num_elements && element > elements[pos] &&
         element == elements[pos + 1]) ||
        (pos + 1 >= 0 && pos + 1 < num_elements &&
         element > elements[pos + 1])) {
      num_errors++;
      std::cout << i << " Error " << element << ": pos = " << pos << " ..."
                << (pos <= 0 ? -1 : elements[pos - 1]) << " , "
                << (pos < 0 || pos >= num_elements ? -1 : elements[pos])
                << " , " << (pos + 1 >= num_elements ? -1 : elements[pos + 1])
                << std::endl;
    }
  }

  std::cout << "Testing BinarySearch_RightMost" << std::endl;
  for (int i = 0; i < num_tests; i++) {
    int element = (rand() % (element_range + 2)) + min_element - 1;
    int pos = gunrock::util::BinarySearch_RightMost(element, elements, 0,
                                                    num_elements - 1);

    // if (element < min_element ||
    //    element >= min_element + element_range ||
    //    pos < 0 ||
    //    pos >= num_elements)
    //    std::cout << "Edge " << element << ": pos = " << pos << " ..."
    //        << (pos <= 0 ? -1 : elements[pos - 1]) << " , "
    //        << (pos < 0 || pos >= num_elements ? -1 : elements[pos]) << " , "
    //        << (pos + 1 >= num_elements ? -1 : elements[pos + 1]) <<
    //        std::endl;

    if ((pos - 1 >= 0 && pos - 1 < num_elements &&
         element < elements[pos - 1]) ||
        (pos >= 0 && pos < num_elements && element < elements[pos]) ||
        (pos + 1 >= 0 && pos + 1 < num_elements &&
         element >= elements[pos + 1])) {
      num_errors++;
      std::cout << i << " Error " << element << ": pos = " << pos << " ..."
                << (pos <= 0 ? -1 : elements[pos - 1]) << " , "
                << (pos < 0 || pos >= num_elements ? -1 : elements[pos])
                << " , " << (pos + 1 >= num_elements ? -1 : elements[pos + 1])
                << std::endl;
    }
  }

  if (num_errors > 0) retval = cudaErrorUnknown;

  delete[] elements;
  elements = NULL;

  return retval;
}
