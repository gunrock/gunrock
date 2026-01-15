# begin /* Set the application name. */
set(APPLICATION_NAME dawn)
# end /* Set the application name. */

# begin /* Add CUDA/HIP executables */
# Use the standard add_essentials_example function for consistency
add_essentials_example(${APPLICATION_NAME})
# end /* Add CUDA/HIP executables */