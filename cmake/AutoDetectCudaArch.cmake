if(NOT DEFINED CUDA_ARCHS)
	############################### Autodetect CUDA Arch #####################################################
	#Auto-detect cuda arch. Inspired by https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
	# This will define and populates CUDA_ARCHS and put it in the cache 
	#Windows users (specially on VS2017 and VS2015) might need to run this 
	#>> "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
	# and change Enterprise to the right edition. More about this here https://stackoverflow.com/a/47746461/1608232
	if(CUDA_FOUND)
		set(cuda_arch_autodetect_file ${CMAKE_BINARY_DIR}/autodetect_cuda_archs.cu)		
		
		file(WRITE ${cuda_arch_autodetect_file} ""
		"#include <stdio.h>\n"
		"int main() {\n"
		"	int count = 0; \n"
		"	if (cudaSuccess != cudaGetDeviceCount(&count)) { return -1; }\n"
		"	if (count == 0) { return -1; }\n"
		"	for (int device = 0; device < count; ++device) {\n"
		"		cudaDeviceProp prop; \n"
		"		bool is_unique = true; \n"
		"		if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
		"			for (int device_1 = device - 1; device_1 >= 0; --device_1) {\n"
		"				cudaDeviceProp prop_1; \n"
		"				if (cudaSuccess == cudaGetDeviceProperties(&prop_1, device_1)) {\n"
		"					if (prop.major == prop_1.major && prop.minor == prop_1.minor) {\n"
		"						is_unique = false; \n"
		"						break; \n"
		"					}\n"
		"				}\n"
		"				else { return -1; }\n"
		"			}\n"
		"			if (is_unique) {\n"
		"				fprintf(stderr, \"-gencode=arch=compute_%d%d,code=sm_%d%d;\", prop.major, prop.minor, prop.major, prop.minor);\n"
		"				fprintf(stderr, \"-gencode=arch=compute_%d%d,code=compute_%d%d;\", prop.major, prop.minor, prop.major, prop.minor);\n"
		"			}\n"
		"		}\n"
		"	}\n"
		"	return 0; \n"
		"}\n")	
	
		execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "-ccbin=${CMAKE_C_COMPILER}" "--run" "${cuda_arch_autodetect_file}"
							#WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/CMakeFiles/"	
							RESULT_VARIABLE CUDA_RETURN_CODE	
							OUTPUT_VARIABLE dummy
							ERROR_VARIABLE fprintf_output					
							OUTPUT_STRIP_TRAILING_WHITESPACE)							

		if(CUDA_RETURN_CODE EQUAL 0)			
			set(CUDA_ARCHS ${fprintf_output} CACHE STRING "CUDA Arch")
		else()
				message(STATUS "GPU architectures auto-detect failed. Will build for all possible architectures.")      
				set(CUDA_ARCHS -gencode=arch=compute_30,code=sm_30
				               -gencode=arch=compute_35,code=sm_35
				               -gencode=arch=compute_37,code=sm_37
				               -gencode=arch=compute_50,code=sm_50
				               -gencode=arch=compute_52,code=sm_52
				               -gencode=arch=compute_60,code=sm_60
				               -gencode=arch=compute_61,code=sm_61
				               -gencode=arch=compute_70,code=sm_70
				               -gencode=arch=compute_72,code=sm_72
				               -gencode=arch=compute_75,code=sm_75
				               -gencode=arch=compute_80,code=sm_80
							   CACHE STRING "CUDA Arch")
		endif()
		message(STATUS "CUDA Autodetected, setting CUDA_ARCHS=" ${CUDA_ARCHS})	
	endif()
endif()
###################################################################################
