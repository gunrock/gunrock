#
# - Finds the Boost headers only
# Use this module by invoking find_package with the form:
#  find_package(Boost
#    [version] [EXACT]      # Minimum or EXACT version e.g. 1.36.0
#    [REQUIRED]             # Fail with error if Boost is not found
#    )
#
# This module finds boost headers the search for the boost headers results
# reported in variables:
#
#  Boost_FOUND            - True if headers and requested libraries were found
#  Boost_INCLUDE_DIRS     - Boost include directories
#  Boost_VERSION          - BOOST_VERSION value from boost/version.hpp
#  Boost_MAJOR_VERSION    - Boost major version number (X in X.y.z)
#  Boost_MINOR_VERSION    - Boost minor version number (Y in x.Y.z)
#  Boost_SUBMINOR_VERSION - Boost subminor version number (Z in x.y.Z)
#
# This module reads hints about search locations from variables:
#  BOOST_INCLUDEDIR       - Preferred include directory e.g. <prefix>/include
#  Boost_NO_SYSTEM_PATHS  - Set to ON to disable searching in locations not
#                           specified by these hint variables. Default is OFF.
#  Boost_ADDITIONAL_VERSIONS
#                         - List of Boost versions not known to this module
#                           (Boost install locations may contain the version)
# and saves search results persistently in CMake cache entries:
#  Boost_INCLUDE_DIR         - Directory containing Boost headers
#
# Users may set these hints or results as cache entries.  Projects should
# not read these entries directly but instead use the above result variables.
# Note that some hint names start in upper-case "BOOST".  One may specify
# these as environment variables if they are not specified as CMake variables
# or cache entries.
#
# This module first searches for the Boost header files using the above hint
# variables  and saves the result in
# Boost_INCLUDE_DIR.
#
#
#=============================================================================
# Copyright 2006-2012 Kitware, Inc.
# Copyright 2006-2008 Andreas Schneider <mail@cynapses.org>
# Copyright 2007      Wengo
# Copyright 2007      Mike Jackson
# Copyright 2008      Andreas Pakulat <apaku@gmx.de>
# Copyright 2008-2012 Philip Lowman <philip@yhbt.com>
#
# CMake - Cross Platform Makefile Generator
# Copyright 2000-2011 Kitware, Inc., Insight Software Consortium
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the names of Kitware, Inc., the Insight Software Consortium,
#   nor the names of their contributors may be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ------------------------------------------------------------------------------
#
# The above copyright and license notice applies to distributions of
# CMake in source and binary form.  Some source files contain additional
# notices of original copyright by their contributors; see each source
# for details.  Third-party software packages supplied with CMake under
# compatible licenses provide their own copyright notices documented in
# corresponding subdirectories.
#
# ------------------------------------------------------------------------------
#
# CMake was initially developed by Kitware with the following sponsorship:
#
#  * National Library of Medicine at the National Institutes of Health
#    as part of the Insight Segmentation and Registration Toolkit (ITK).
#
#  * US National Labs (Los Alamos, Livermore, Sandia) ASC Parallel
#    Visualization Initiative.
#
#  * National Alliance for Medical Image Computing (NAMIC) is funded by the
#    National Institutes of Health through the NIH Roadmap for Medical Research,
#    Grant U54 EB005149.
#
#  * Kitware, Inc.
#
#
#

#-------------------------------------------------------------------------------
#  FindBoost functions & macros
#
macro(_Boost_CHANGE_DETECT changed_var)
  set(${changed_var} 0)
  foreach(v ${ARGN})
    if(DEFINED _Boost_COMPONENTS_SEARCHED)
      if(${v})
        if(_${v}_LAST)
          string(COMPARE NOTEQUAL "${${v}}" "${_${v}_LAST}" _${v}_CHANGED)
        else()
          set(_${v}_CHANGED 1)
        endif()
      elseif(_${v}_LAST)
        set(_${v}_CHANGED 1)
      endif()
      if(_${v}_CHANGED)
        set(${changed_var} 1)
      endif()
    else()
      set(_${v}_CHANGED 0)
    endif()
  endforeach()
endmacro()

#
# Make sure the spelling of boost options is correct
#
function(_Boost_CHECK_SPELLING _var)
  if(${_var})
    string(TOUPPER ${_var} _var_UC)
    message(FATAL_ERROR "ERROR: ${_var} is not the correct spelling.  The proper spelling is ${_var_UC}.")
  endif()
endfunction()

#
# End functions/macros
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# main.
#-------------------------------------------------------------------------------

# Check the version of Boost against the requested version.
if(Boost_FIND_VERSION AND NOT Boost_FIND_VERSION_MINOR)
  message(SEND_ERROR "When requesting a specific version of Boost, you must provide at least the major and minor version numbers, e.g., 1.34")
endif()

if(Boost_FIND_VERSION_EXACT)
  # The version may appear in a directory with or without the patch
  # level, even when the patch level is non-zero.
  set(_boost_TEST_VERSIONS
    "${Boost_FIND_VERSION_MAJOR}.${Boost_FIND_VERSION_MINOR}.${Boost_FIND_VERSION_PATCH}"
    "${Boost_FIND_VERSION_MAJOR}.${Boost_FIND_VERSION_MINOR}")
else()
  # The user has not requested an exact version.  Among known
  # versions, find those that are acceptable to the user request.
  set(_Boost_KNOWN_VERSIONS ${Boost_ADDITIONAL_VERSIONS}
    "1.56.0" "1.56" "1.55.0" "1.55" "1.54.0" "1.54"
    "1.53.0" "1.53" "1.52.0" "1.52" "1.51.0" "1.51"
    "1.50.0" "1.50" "1.49.0" "1.49" "1.48.0" "1.48" "1.47.0" "1.47" "1.46.1"
    "1.46.0" "1.46" "1.45.0" "1.45" "1.44.0" "1.44" "1.43.0" "1.43" "1.42.0" "1.42"
    "1.41.0" "1.41" "1.40.0" "1.40" "1.39.0" "1.39" "1.38.0" "1.38" "1.37.0" "1.37"
    "1.36.1" "1.36.0" "1.36" "1.35.1" "1.35.0" "1.35" "1.34.1" "1.34.0"
    "1.34" "1.33.1" "1.33.0" "1.33")
  set(_boost_TEST_VERSIONS)
  if(Boost_FIND_VERSION)
    set(_Boost_FIND_VERSION_SHORT "${Boost_FIND_VERSION_MAJOR}.${Boost_FIND_VERSION_MINOR}")
    # Select acceptable versions.
    foreach(version ${_Boost_KNOWN_VERSIONS})
      if(NOT "${version}" VERSION_LESS "${Boost_FIND_VERSION}")
        # This version is high enough.
        list(APPEND _boost_TEST_VERSIONS "${version}")
      elseif("${version}.99" VERSION_EQUAL "${_Boost_FIND_VERSION_SHORT}.99")
        # This version is a short-form for the requested version with
        # the patch level dropped.
        list(APPEND _boost_TEST_VERSIONS "${version}")
      endif()
    endforeach()
  else()
    # Any version is acceptable.
    set(_boost_TEST_VERSIONS "${_Boost_KNOWN_VERSIONS}")
  endif()
endif()

# The reason that we failed to find Boost. This will be set to a
# user-friendly message when we fail to find some necessary piece of
# Boost.
set(Boost_ERROR_REASON)

if(Boost_DEBUG)
  # Output some of their choices
  message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                 "_boost_TEST_VERSIONS = ${_boost_TEST_VERSIONS}")
  message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                 "Boost_ADDITIONAL_VERSIONS = ${Boost_ADDITIONAL_VERSIONS}")
  message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                 "Boost_NO_SYSTEM_PATHS = ${Boost_NO_SYSTEM_PATHS}")
endif()

_Boost_CHECK_SPELLING(Boost_INCLUDEDIR)

# Collect environment variable inputs as hints.  Do not consider changes.
foreach(v BOOST_INCLUDEDIR)
  set(_env $ENV{${v}})
  if(_env)
    file(TO_CMAKE_PATH "${_env}" _ENV_${v})
  else()
    set(_ENV_${v} "")
  endif()
endforeach()

# Collect inputs and cached results.  Detect changes since the last run.
set(_Boost_VARS_DIR
  Boost_NO_SYSTEM_PATHS
  )

if(Boost_DEBUG)
  message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                 "Declared as CMake or Environmental Variables:")
  message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                 "  BOOST_INCLUDEDIR = ${BOOST_INCLUDEDIR}")
  message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                 "_boost_TEST_VERSIONS = ${_boost_TEST_VERSIONS}")
endif()

# ------------------------------------------------------------------------
#  Search for Boost include DIR
# ------------------------------------------------------------------------

set(_Boost_VARS_INC BOOST_INCLUDEDIR Boost_INCLUDE_DIR Boost_ADDITIONAL_VERSIONS)
_Boost_CHANGE_DETECT(_Boost_CHANGE_INCDIR ${_Boost_VARS_DIR} ${_Boost_VARS_INC})
# Clear Boost_INCLUDE_DIR if it did not change but other input affecting the
# location did.  We will find a new one based on the new inputs.
if(_Boost_CHANGE_INCDIR AND NOT _Boost_INCLUDE_DIR_CHANGED)
  unset(Boost_INCLUDE_DIR CACHE)
endif()

if(NOT Boost_INCLUDE_DIR)
  set(_boost_INCLUDE_SEARCH_DIRS "")
  if(BOOST_INCLUDEDIR)
    list(APPEND _boost_INCLUDE_SEARCH_DIRS ${BOOST_INCLUDEDIR})
  elseif(_ENV_BOOST_INCLUDEDIR)
    list(APPEND _boost_INCLUDE_SEARCH_DIRS ${_ENV_BOOST_INCLUDEDIR})
  endif()

  if( Boost_NO_SYSTEM_PATHS)
    list(APPEND _boost_INCLUDE_SEARCH_DIRS NO_CMAKE_SYSTEM_PATH)
  else()
    list(APPEND _boost_INCLUDE_SEARCH_DIRS PATHS
      C:/boost/include
      C:/boost
      /sw/local/include
      )
  endif()

  # Try to find Boost by stepping backwards through the Boost versions
  # we know about.
  # Build a list of path suffixes for each version.
  set(_boost_PATH_SUFFIXES)
  foreach(_boost_VER ${_boost_TEST_VERSIONS})
    # Add in a path suffix, based on the required version, ideally
    # we could read this from version.hpp, but for that to work we'd
    # need to know the include dir already
    set(_boost_BOOSTIFIED_VERSION)

    # Transform 1.35 => 1_35 and 1.36.0 => 1_36_0
    if(_boost_VER MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\1_\\2_\\3"
          _boost_BOOSTIFIED_VERSION ${_boost_VER})
    elseif(_boost_VER MATCHES "[0-9]+\\.[0-9]+")
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+)" "\\1_\\2"
          _boost_BOOSTIFIED_VERSION ${_boost_VER})
    endif()

    list(APPEND _boost_PATH_SUFFIXES
      "boost-${_boost_BOOSTIFIED_VERSION}"
      "boost_${_boost_BOOSTIFIED_VERSION}"
      "boost/boost-${_boost_BOOSTIFIED_VERSION}"
      "boost/boost_${_boost_BOOSTIFIED_VERSION}"
      )

  endforeach()

  if(Boost_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "Include debugging info:")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "  _boost_INCLUDE_SEARCH_DIRS = ${_boost_INCLUDE_SEARCH_DIRS}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "  _boost_PATH_SUFFIXES = ${_boost_PATH_SUFFIXES}")
  endif()

  # Look for a standard boost header file.
  find_path(Boost_INCLUDE_DIR
    NAMES         boost/config.hpp
    HINTS         ${_boost_INCLUDE_SEARCH_DIRS}
    PATH_SUFFIXES ${_boost_PATH_SUFFIXES}
    )
endif()

# ------------------------------------------------------------------------
#  Extract version information from version.hpp
# ------------------------------------------------------------------------

# Set Boost_FOUND based only on header location and version.
# It will be updated below for component libraries.
if(Boost_INCLUDE_DIR)
  if(Boost_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of version.hpp: ${Boost_INCLUDE_DIR}/boost/version.hpp")
  endif()

  # Extract Boost_VERSION and Boost_LIB_VERSION from version.hpp
  set(Boost_VERSION 0)
  set(Boost_LIB_VERSION "")
  file(STRINGS "${Boost_INCLUDE_DIR}/boost/version.hpp" _boost_VERSION_HPP_CONTENTS REGEX "#define BOOST_(LIB_)?VERSION ")
  set(_Boost_VERSION_REGEX "([0-9]+)")
  set(_Boost_LIB_VERSION_REGEX "\"([0-9_]+)\"")
  foreach(v VERSION LIB_VERSION)
    if("${_boost_VERSION_HPP_CONTENTS}" MATCHES ".*#define BOOST_${v} ${_Boost_${v}_REGEX}.*")
      set(Boost_${v} "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  unset(_boost_VERSION_HPP_CONTENTS)

  math(EXPR Boost_MAJOR_VERSION "${Boost_VERSION} / 100000")
  math(EXPR Boost_MINOR_VERSION "${Boost_VERSION} / 100 % 1000")
  math(EXPR Boost_SUBMINOR_VERSION "${Boost_VERSION} % 100")

  set(Boost_ERROR_REASON
    "${Boost_ERROR_REASON}Boost version: ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}\nBoost include path: ${Boost_INCLUDE_DIR}")
  if(Boost_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "version.hpp reveals boost "
                   "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
  endif()

  if(Boost_FIND_VERSION)
    # Set Boost_FOUND based on requested version.
    set(_Boost_VERSION "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
    if("${_Boost_VERSION}" VERSION_LESS "${Boost_FIND_VERSION}")
      set(Boost_FOUND 0)
      set(_Boost_VERSION_AGE "old")
    elseif(Boost_FIND_VERSION_EXACT AND
        NOT "${_Boost_VERSION}" VERSION_EQUAL "${Boost_FIND_VERSION}")
      set(Boost_FOUND 0)
      set(_Boost_VERSION_AGE "new")
    else()
      set(Boost_FOUND 1)
    endif()
    if(NOT Boost_FOUND)
      # State that we found a version of Boost that is too new or too old.
      set(Boost_ERROR_REASON
        "${Boost_ERROR_REASON}\nDetected version of Boost is too ${_Boost_VERSION_AGE}. Requested version was ${Boost_FIND_VERSION_MAJOR}.${Boost_FIND_VERSION_MINOR}")
      if (Boost_FIND_VERSION_PATCH)
        set(Boost_ERROR_REASON
          "${Boost_ERROR_REASON}.${Boost_FIND_VERSION_PATCH}")
      endif ()
      if (NOT Boost_FIND_VERSION_EXACT)
        set(Boost_ERROR_REASON "${Boost_ERROR_REASON} (or newer)")
      endif ()
      set(Boost_ERROR_REASON "${Boost_ERROR_REASON}.")
    endif ()
  else()
    # Caller will accept any Boost version.
    set(Boost_FOUND 1)
  endif()
else()
  set(Boost_FOUND 0)
  set(Boost_ERROR_REASON
    "${Boost_ERROR_REASON}Unable to find the Boost header files. Please set BOOST_INCLUDEDIR to the directory containing Boost's headers.")
endif()

set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})
set(Boost_LIBRARY_DIRS ${Boost_LIBRARY_DIR})

# ------------------------------------------------------------------------
#  Notification to end user about what was found
# ------------------------------------------------------------------------

if(Boost_FOUND)
  if(NOT Boost_FIND_QUIETLY)
    message(STATUS "Boost version: ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
  endif()
else()
  if(Boost_FIND_REQUIRED)
    message(SEND_ERROR "Unable to find the requested Boost libraries.\n${Boost_ERROR_REASON}")
  else()
    if(NOT Boost_FIND_QUIETLY)
      # we opt not to automatically output Boost_ERROR_REASON here as
      # it could be quite lengthy and somewhat imposing in its requests
      # Since Boost is not always a required dependency we'll leave this
      # up to the end-user.
      if(Boost_DEBUG OR Boost_DETAILED_FAILURE_MSG)
        message(STATUS "Could NOT find Boost\n${Boost_ERROR_REASON}")
      else()
        message(STATUS "Could NOT find Boost")
      endif()
    endif()
  endif()
endif()

# Configure display of cache entries in GUI.
foreach(v ${_Boost_VARS_INC} ${_Boost_VARS_LIB})
  get_property(_type CACHE ${v} PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      if("x${v}" STREQUAL "xBoost_ADDITIONAL_VERSIONS")
        set_property(CACHE ${v} PROPERTY TYPE STRING)
      else()
        set_property(CACHE ${v} PROPERTY TYPE PATH)
      endif()
    endif()
  endif()
endforeach()

# Record last used values of input variables so we can
# detect on the next run if the user changed them.
foreach(v
    ${_Boost_VARS_INC} ${_Boost_VARS_LIB}
    ${_Boost_VARS_DIR} ${_Boost_VARS_NAME}
    )
  if(DEFINED ${v})
    set(_${v}_LAST "${${v}}" CACHE INTERNAL "Last used ${v} value.")
  else()
    unset(_${v}_LAST CACHE)
  endif()
endforeach()