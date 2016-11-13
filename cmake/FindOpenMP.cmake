# ------------------------------------------------------------------------
#  Gunrock: Find OpenMP includes and libraries
# ------------------------------------------------------------------------

IF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # also see how to do overrides:
  # https://gist.github.com/robertmaynard/11297565

  # updated for CUDA 7, which uses libc++ rather than libstdc++
  # older settings: check the git history
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} 	 /opt/local/lib /opt/local/lib/libomp)
  SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} 	 /opt/local/include /opt/local/include/libomp)

  IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    #added for OpenMP on OS X using Macports
    FIND_PATH(OPENMP_INCLUDE omp.h PATHS ${CMAKE_INCLUDE_PATH})
    FIND_LIBRARY(OPENMP_LIB NAMES libgomp.dylib libomp.dylib libiomp5.dylib PATHS ${CMAKE_LIBRARY_PATH})
    IF (OPENMP_INCLUDE OR OPENMP_LIB)

      link_directories("/opt/local/lib" "/opt/local/lib/libomp")
      include_directories("/opt/local/include" "/opt/local/include/libomp")

      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      MESSAGE(STATUS "Found OpenMP (-libomp)")
    ELSE()
      MESSAGE(WARNING "OpenMP (-libomp) was requested but support was not found")
    ENDIF(OPENMP_INCLUDE OR OPENMP_LIB)

  ENDIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

ELSE()

  #added for OpenMP on Linux
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    MESSAGE(STATUS "Found OpenMP")
  ELSE()
    MESSAGE(WARNING "OpenMP was requested but support was not found")
  ENDIF()
ENDIF (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
