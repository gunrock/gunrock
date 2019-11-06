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
    IF (NOT OPENMP_INCLUDE OR NOT OPENMP_LIB)
      # If OpenMP doesn't exist, cmake automatically build and install it on Mac OS X.
      INCLUDE(ExternalProject)
      SET(OPENMP_SOURCES_DIR ${CMAKE_BINARY_DIR}/omp)
      SET(OPENMP_INSTALL_DIR ${CMAKE_BINARY_DIR}/install/omp)
      SET(OPENMP_INCLUDE "${OPENMP_INSTALL_DIR}/include" CACHE PATH "openmp include directory." FORCE)
      SET(OPENMP_LIBRARY "${OPENMP_INSTALL_DIR}/lib/libomp.dylib" CACHE FILEPATH "openmp library." FORCE)

      # external dependencies log output
      SET(EXTERNAL_PROJECT_LOG_ARGS
          LOG_DOWNLOAD    0     # Wrap download in script to log output
          LOG_UPDATE      1     # Wrap update in script to log output
          LOG_CONFIGURE   1     # Wrap configure in script to log output
          LOG_BUILD       0     # Wrap build in script to log output
          LOG_TEST        1     # Wrap test in script to log output
          LOG_INSTALL     0     # Wrap install in script to log output
      )

      ExternalProject_Add(
        extern_openmp
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY  https://github.com/llvm-mirror/openmp.git
        GIT_TAG         1d9902d5b99ee93e0bb3f886e08414a81f21cd2a
        PREFIX          ${OPENMP_SOURCES_DIR}
        UPDATE_COMMAND  ""
        CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                        -DCMAKE_INSTALL_PREFIX=${OPENMP_INSTALL_DIR}
                        -DCMAKE_INSTALL_LIBDIR=${OPENMP_INSTALL_DIR}/lib
                        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                        -DBUILD_TESTING=OFF
        CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${OPENMP_INSTALL_DIR}
                         -DCMAKE_INSTALL_LIBDIR:PATH=${OPENMP_INSTALL_DIR}/lib
                         -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      )   
      ADD_LIBRARY(openmp SHARED IMPORTED GLOBAL)
      SET_PROPERTY(TARGET openmp PROPERTY IMPORTED_LOCATION ${OPENMP_LIBRARY})
      ADD_DEPENDENCIES(openmp extern_openmp)
      SET(OPENMP_LIB openmp)
    ENDIF()

    LINK_LIBRARIES("${OPENMP_LIB}")
    INCLUDE_DIRECTORIES("${OPENMP_INCLUDE}")
    MESSAGE(STATUS "Found OpenMP (-libomp)")
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
