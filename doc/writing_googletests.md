How to write GoogleTest for Gunrock? {#googletestgunrock}
================

**Recommended Read:** [Introduction: Why Google C++ Testing Framework?](https://github.com/google/googletest/blob/master/googletest/docs/Primer.md)

When writing a good test, we would like to cover all possible functions (or execute all code lines),
what I will recommend to do is write a simple test, run code coverage on it, and
use codecov.io to determine what lines are not executed. This gives you a good
idea of what needs to be in the test and what you are missing.

**What is code coverage?**
> Code coverage is a measurement used to express which lines of code were executed by a test suite. We use three primary terms to describe each lines executed.
>
> * hit indicates that the source code was executed by the test suite.
> * partial indicates that the source code was not fully executed by the test suite; there are remaining branches that were not executed.
> * miss indicates that the source code was not executed by the test suite.
>
> Coverage is the ratio of hits / (hit + partial + miss). A code base that has 5 lines executed by tests out of 12 total lines will receive a coverage ratio of 41% (rounding down).

Below is an example of what lines are a hit and a miss; you can target the lines missed in the tests to improve coverage.

![Example CodeCov Stats](https://i.imgur.com/5QwKjcB.png)

Example Test Using GoogleTest
================
1. Create a `test_<test-name>.h` file and place it in the appropriate directory inside `/path/to/gunrock/tests/`. I will be using `test_bfs_lib.h` as an example.

2. In the `tests/test.cpp` file, add your test file as an include:
```C
// Add google tests
#include "bfs/test_lib_bfs.h"
```

3. In your `test_<test-name>.h` file, create a `TEST()` function, which takes two parameters: `TEST(<nameofthesuite>, <nameofthetest>)`.
4. Use `EXPECT` and `ASSERT` to write the actual test itself. I have provided a commented example below:

```C
/**
 * @brief BFS test for shared library advanced interface
 * @file test_lib_bfs.h
 */

// Includes required for the test
#include <stdio.h>
#include <gunrock/gunrock.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Add to gunrock's namespace
namespace gunrock {

/* Test function, test suite in this case is
 * sharedlibrary and the test itself is breadthfirstsearch
 */
TEST(sharedlibrary, breadthfirstsearch)
{
    struct GRTypes data_t;                 // data type structure
    data_t.VTXID_TYPE = VTXID_INT;         // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;         // graph size type
    data_t.VALUE_TYPE = VALUE_INT;         // attributes type
    int srcs[3] = {0,1,2};

    struct GRSetup *config = InitSetup(3, srcs);   // gunrock configurations

    int num_nodes = 7, num_edges = 15;  // number of nodes and edges
    int row_offsets[8]  = {0, 3, 6, 9, 11, 14, 15, 15};
    int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graphi->num_nodes   = num_nodes;
    graphi->num_edges   = num_edges;
    graphi->row_offsets = (void*)&row_offsets[0];
    graphi->col_indices = (void*)&col_indices[0];

    gunrock_bfs(grapho, graphi, config, data_t);

    int *labels = (int*)malloc(sizeof(int) * graphi->num_nodes);
    labels = (int*)grapho->node_value1;

    // IMPORTANT: Expected output is stored in an array to compare against determining if the test passed or failed
    int result[7] = {2147483647, 2147483647, 0, 1, 1, 1, 2};

    for (int i = 0; i < graphi->num_nodes; ++i) {
      // IMPORTANT: Compare expected result with the generated labels
      EXPECT_EQ(labels[i], result[i]) << "Vectors x and y differ at index " << i;
    }

    if (graphi) free(graphi);
    if (grapho) free(grapho);
    if (labels) free(labels);

}
} // namespace gunrock
```

5. Now when you run the binary called `unit_test`, it will automatically run your test suite along with all other google tests as well.
This binary it automatically compiled when gunrock is built, and is found in `/path/to/builddir/bin/unit_test`.

**Final Remarks:**

* I highly recommend reading the Primer document mentioned at the start of this tutorial. It explains in detail how to write a unit test using google test. My tutorial has more been about how to incorporate it into Gunrock.
* Another interesting read is [Measuring Coverage at Google](https://testing.googleblog.com/2014/07/measuring-coverage-at-google.html).
