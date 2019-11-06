// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * misc_utils.cuh
 *
 * @brief Misc. Utility Routines (header)
 */

#ifndef MISC_UTILS_H
#define MISC_UTILS_H

// Pthread code based on:
// http://blog.albertarmea.com/post/47089939939/using-pthread-barrier-on-mac-os-x
// Apple/clang does not support these pthread calls. Thus they will
// only be included and linked if it's Apple/clang.

#include <pthread.h>
#include <errno.h>

#ifdef __APPLE__
#ifdef __clang__
typedef int pthread_barrierattr_t;
typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int count;
  int tripCount;
} pthread_barrier_t;

int pthread_barrier_init(pthread_barrier_t *barrier,
                         const pthread_barrierattr_t *attr, unsigned int count);
int pthread_barrier_destroy(pthread_barrier_t *barrier);
int pthread_barrier_wait(pthread_barrier_t *barrier);
#endif
#endif

#endif  // MISC_UTILS_H
