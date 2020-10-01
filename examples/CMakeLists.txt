# Gunrock's Testing CMake File

# begin /* Add premitives' subdirectories */
if(GUNROCK_BUILD_APPLICATIONS)
  #add_subdirectory(astar)
  add_subdirectory(bc)
  add_subdirectory(bfs)
  #add_subdirectory(cc)
  add_subdirectory(color)
  add_subdirectory(geo)
  #add_subdirectory(grmat)
  ###add_subdirectory(gtf)
  add_subdirectory(hits)
  add_subdirectory(knn)
  add_subdirectory(louvain)
  #add_subdirectory(lp)
  ###add_subdirectory(mf)
  #add_subdirectory(mst)
  add_subdirectory(pr)
  add_subdirectory(pr_nibble)
  add_subdirectory(proj)
  add_subdirectory(rw)
  add_subdirectory(sage)
  #add_subdirectory(salsa)
  add_subdirectory(sm)
  add_subdirectory(snn)
  add_subdirectory(ss)
  add_subdirectory(sssp)
  add_subdirectory(tc)
  #add_subdirectory(topk)
  add_subdirectory(vn)
  add_subdirectory(lp)
  #add_subdirectory(wtf)

  # TUTORIAL SUBDIRECTORIES
  #add_subdirectory(gtc)
  #add_subdirectory(hello)

  # SHARED LIBRARY SUBDIRECTORY
  add_subdirectory(shared_libs)

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

  if(GUNROCK_APP_COLOR)
    add_subdirectory(color)
  endif(GUNROCK_APP_COLOR)

  if(GUNROCK_APP_GEO)
    add_subdirectory(geo)
  endif(GUNROCK_APP_GEO)

  if(GUNROCK_APP_GTF)
    add_subdirectory(gtf)
  endif(GUNROCK_APP_GTF)

  if(GUNROCK_APP_HITS)
    add_subdirectory(hits)
  endif(GUNROCK_APP_HITS)

  if(GUNROCK_APP_KNN)
    add_subdirectory(knn)
  endif(GUNROCK_APP_KNN)

  if(GUNROCK_APP_LOUVAIN)
    add_subdirectory(louvain)
  endif(GUNROCK_APP_LOUVAIN)

  if(GUNROCK_APP_LP)
    add_subdirectory(lp)
  endif(GUNROCK_APP_LP)

  if(GUNROCK_APP_MF)
    add_subdirectory(mf)
  endif(GUNROCK_APP_MF)

  if(GUNROCK_APP_MST)
    add_subdirectory(mst)
  endif(GUNROCK_APP_MST)

  if(GUNROCK_APP_PR)
    add_subdirectory(pr)
  endif(GUNROCK_APP_PR)

  if(GUNROCK_APP_PR_NIBBLE)
    add_subdirectory(pr_nibble)
  endif(GUNROCK_APP_PR_NIBBLE)

  if(GUNROCK_APP_PROJ)
    add_subdirectory(proj)
  endif(GUNROCK_APP_PROJ)

  if(GUNROCK_APP_RW)
    add_subdirectory(rw)
  endif(GUNROCK_APP_RW)

  if(GUNROCK_APP_SAGE)
    add_subdirectory(sage)
  endif(GUNROCK_APP_SAGE)

  if(GUNROCK_APP_SALSA)
    add_subdirectory(salsa)
  endif(GUNROCK_APP_SALSA)

  if(GUNROCK_APP_SM)
    add_subdirectory(sm)
  endif(GUNROCK_APP_SM)

  if(GUNROCK_APP_SNN)
    add_subdirectory(snn)
  endif(GUNROCK_APP_SNN)

  if(GUNROCK_APP_SS)
    add_subdirectory(ss)
  endif(GUNROCK_APP_SS)

  if(GUNROCK_APP_SSSP)
    add_subdirectory(sssp)
  endif(GUNROCK_APP_SSSP)

  if(GUNROCK_APP_TC)
    add_subdirectory(tc)
  endif(GUNROCK_APP_TC)

  if(GUNROCK_APP_TOPK)
    add_subdirectory(topk)
  endif(GUNROCK_APP_TOPK)

  if(GUNROCK_APP_VN)
    add_subdirectory(vn)
  endif(GUNROCK_APP_VN)

  if(GUNROCK_APP_WTF)
    add_subdirectory(wtf)
  endif(GUNROCK_APP_WTF)

  # if(GUNROCK_APP_HELLO)
  #   add_subdirectory(hello)
  # endif(GUNROCK_APP_HELLO)

  # if(GUNROCK_APP_GTC)
  #   add_subdirectory(gtc)
  # endif(GUNROCK_APP_GTC)

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
