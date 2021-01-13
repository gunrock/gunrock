# ------------------------------------------------------------------------
#  Find external moderngpu directories
# ------------------------------------------------------------------------
SET(MODERNGPU_INCLUDEDIR
  ${PROJECT_SOURCE_DIR}/externals/moderngpu/src
  CACHE PATH
  "Directory to the Modern GPU include files")

SET(MODERNGPU_SOURCE_DIR
  ${PROJECT_SOURCE_DIR}/externals/moderngpu/src/moderngpu
  CACHE PATH
  "Directory to the Modern GPU source files")

find_path(
  MODERNGPU_INCLUDEDIR
  NAMES moderngpu/context.hxx
  PATHS ${MODERNGPU_INCLUDE_DIR}
  DOC "Include directory for the moderngpu library."
)

mark_as_advanced(MODERNGPU_INCLUDE_DIR)

if(MODERNGPU_INCLUDE_DIR)
  set(MODERNGPU_FOUND TRUE)
endif()

mark_as_advanced(MODERNGPU_FOUND)