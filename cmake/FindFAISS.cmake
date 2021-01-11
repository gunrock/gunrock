# ------------------------------------------------------------------------
#  Gunrock: Find FAISS directories
# ------------------------------------------------------------------------
FIND_LIBRARY( FAISS_LIBRARY
              NAMES faiss
              PATHS ../faiss /lib /usr/lib /usr/lib64 /usr/local/lib)

FIND_PATH(  FAISS_INCLUDE_DIR
            NAMES faiss/gpu/GpuIndexFlat.h
            PATHS /usr/include ../)

IF (FAISS_LIBRARY)
    ADD_DEFINITIONS( -DFAISS_FOUND=1)
ENDIF (FAISS_LIBRARY)
