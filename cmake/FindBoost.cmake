# ------------------------------------------------------------------------
#  Gunrock: Finds Boost package and includes/links directories
# ------------------------------------------------------------------------
SET(gunrock_REQUIRED_BOOST_VERSION 1.58)

FIND_PACKAGE(Boost ${gunrock_REQUIRED_BOOST_VERSION})
IF (Boost_FOUND)
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
  LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
ELSE ()
  MESSAGE(WARNING "Boost was requested but support was not found, run
  `sudo apt-get install libboost1.58-all-dev` `/dep/install_boost.sh` for installation.")
ENDIF ()
