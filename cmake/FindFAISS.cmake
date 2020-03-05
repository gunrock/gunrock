# ------------------------------------------------------------------------
#  Gunrock: Find FAISS directories
# ------------------------------------------------------------------------
FIND_LIBRARY( FAISS_LIBRARY
              NAMES faiss
              PATHS ${CMAKE_SOURCE_DIR}/externals/faiss /lib /usr/lib /usr/lib64 /usr/local/lib)

IF (FAISS_LIBRARY)
    SET(FAISS_FOUND TRUE)
ELSE ()
    SET(FAISS_FOUND FALSE)
ENDIF (FAISS_LIBRARY)
