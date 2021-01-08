# ------------------------------------------------------------------------
#  Find external mtx directories
# ------------------------------------------------------------------------
SET(MTX_INCLUDEDIR
  ${CMAKE_SOURCE_DIR}/externals/mtx
  CACHE PATH
  "Directory to the MTX include files")

SET(MTX_SOURCE_DIR
  ${CMAKE_SOURCE_DIR}/externals/mtx
  CACHE PATH
  "Directory to the MTX source files")

find_path(
  MTX_INCLUDEDIR
  NAMES mmio.hxx
  PATHS ${MTX_INCLUDE_DIR}
  DOC "Include directory for the mtx library."
)

mark_as_advanced(MTX_INCLUDE_DIR)

if(MTX_INCLUDE_DIR)
  set(MTX_FOUND TRUE)
endif()

mark_as_advanced(MTX_FOUND)