# ------------------------------------------------------------------------
#  Gunrock: Find external slabhash directories
# ------------------------------------------------------------------------
SET(slabhash_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/SlabHash/src
  CACHE PATH
  "Directory to the SlabHash include files")

SET(slaballoc_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/externals/SlabHash/SlabAlloc/src
  CACHE PATH
  "Directory to the SlabAlloc include files")
