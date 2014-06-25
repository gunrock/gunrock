/**
 * @file gunrock.h
 * @brief Main Library Header File. Defines Public Interface.
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Gunrock Result codes returned by Gunrock API functions.
 */
enum GunrockResult
{
    GUNROCK_SUCCESS = 0,                 /**< No error. */
    GUNROCK_ERROR_INVALID_HANDLE,        /**< Specified handle (for example, 
                                            to a plan) is invalid. **/
    GUNROCK_ERROR_ILLEGAL_CONFIGURATION, /**< Specified configuration is
                                            illegal. For example, an
                                            invalid or illogical
                                            combination of options. */
    GUNROCK_ERROR_INVALID_PLAN,          /**< The plan is not configured properly.
                                            For example, passing a plan for scan
                                            to GUNROCKSegmentedScan. */
    GUNROCK_ERROR_INSUFFICIENT_RESOURCES,/**< The function could not complete due to
                                            insufficient resources (typically CUDA
                                            device resources such as shared memory)
                                            for the specified problem size. */
    GUNROCK_ERROR_UNKNOWN = 9999         /**< Unknown or untraceable error. */
};

#define GUNROCK_INVALID_HANDLE 0xC0DABAD1
#define GUNROCK_DLL   

typedef size_t GunrockHandle;



// gunrock Initialization
GUNROCK_DLL
GunrockResult gunrockCreate(GunrockHandle* theGunrock);

// gunrock Destruction
GUNROCK_DLL
GunrockResult gunrockDestroy(GunrockHandle theGunrock);

GUNROCK_DLL
GunrockResult gunrock_mad_int(int *origin_elements, int num_elements);

GUNROCK_DLL
GunrockResult gunrock_mad_float(float *origin_elements, int num_elements);

#ifdef __cplusplus
}
#endif
