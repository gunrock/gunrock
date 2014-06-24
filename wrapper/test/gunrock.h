/**
 * @file gunrock.h
 * @brief Main Library Header File. Defines Public Interface.
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

void gunrock_mad_int(int *origin_elements, int num_elements);
void gunrock_mad_float(float *origin_elements, int num_elements);

#ifdef __cplusplus
}
#endif
