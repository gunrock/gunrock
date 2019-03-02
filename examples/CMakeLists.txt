# Gunrock's Testing CMake File

# begin /* Add premitives' subdirectories */
if(GUNROCK_BUILD_APPLICATIONS)
  add_subdirectory(shared_libs)

  add_subdirectory(astar)
  add_subdirectory(bc)
  add_subdirectory(bfs)
  add_subdirectory(cc)
  add_subdirectory(pr)
  add_subdirectory(sssp)
  add_subdirectory(hits)
  add_subdirectory(salsa)
  add_subdirectory(mst)
  add_subdirectory(wtf)
  add_subdirectory(topk)
  add_subdirectory(rw)
  add_subdirectory(lp)
  add_subdirectory(tc)
  add_subdirectory(grmat)

  #add_subdirectory(simple_example)
  #add_subdirectory(template)

# Individual options to build specific applications
else(GUNROCK_BUILD_APPLICATIONS)
  if(GUNROCK_APP_ASTAR)
    add_subdirectory(astar)
  endif(GUNROCK_APP_ASTAR)

  if(GUNROCK_APP_BC)
    add_subdirectory(bc)
  endif(GUNROCK_APP_BC)

  if(GUNROCK_APP_BFS)
    add_subdirectory(bfs)
  endif(GUNROCK_APP_BFS)

  if(GUNROCK_APP_CC)
    add_subdirectory(cc)
  endif(GUNROCK_APP_CC)

  if(GUNROCK_APP_PR)
    add_subdirectory(pr)
  endif(GUNROCK_APP_PR)

  if(GUNROCK_APP_SSSP)
    add_subdirectory(sssp)
  endif(GUNROCK_APP_SSSP)

  if(GUNROCK_APP_HITS)
    add_subdirectory(hits)
  endif(GUNROCK_APP_HITS)

  if(GUNROCK_APP_SALSA)
    add_subdirectory(salsa)
  endif(GUNROCK_APP_SALSA)

  if(GUNROCK_APP_MST)
    add_subdirectory(mst)
  endif(GUNROCK_APP_MST)

  if(GUNROCK_APP_WTF)
    add_subdirectory(wtf)
  endif(GUNROCK_APP_WTF)

  if(GUNROCK_APP_TOPK)
    add_subdirectory(topk)
  endif(GUNROCK_APP_TOPK)

  if(GUNROCK_APP_RW)
    add_subdirectory(rw)
  endif(GUNROCK_APP_RW)

  if(GUNROCK_APP_LP)
    add_subdirectory(lp)
  endif(GUNROCK_APP_LP)

  if(GUNROCK_APP_TC)
    add_subdirectory(tc)
  endif(GUNROCK_APP_TC)

  if(GUNROCK_APP_GRMAT)
    add_subdirectory(grmat)
  endif(GUNROCK_APP_GRMAT)

  # if(GUNROCK_APP_SAMPLE)
  #   add_subdirectory(sample)
  # endif(GUNROCK_APP_SAMPLE)

endif(GUNROCK_BUILD_APPLICATIONS)
# end /* Add premitives' subdirectories */

# note: Some premitives are not added as test because they don't have
#	cpu reference code.

if(GUNROCK_MGPU_TESTS)
  if(DEFINED DEVICES)
    set(DEVICES "--device=${DEVICES}")
  else(DEFINED DEVICES)
    message(WARNING "GUNROCK_MGPU_TESTS was set ON, but devices were not"
		    " specified, using `-DDEVICES=0,0` instead (default).")
    set(DEVICES "--device=0,0")
  endif(DEFINED DEVICES)
endif(GUNROCK_MGPU_TESTS)
# end /* Enable Testing for `ctest` */
