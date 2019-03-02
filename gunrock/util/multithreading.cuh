/* Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef MULTITHREADING_H
#define MULTITHREADING_H

// Simple portable thread library.

#if _WIN32
// Windows threads.
#include <windows.h>

typedef HANDLE CUTThread;
typedef unsigned(WINAPI *CUT_THREADROUTINE)(void *);

struct CUTBarrier {
  CRITICAL_SECTION criticalSection;
  HANDLE barrierEvent;
  int releaseCount;
  int count;
};

#define CUT_THREADPROC unsigned WINAPI
#define CUT_THREADEND return 0

#else
// POSIX threads.
#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void *
#define CUT_THREADEND return 0

struct CUTBarrier {
  pthread_mutex_t mutex;
  pthread_cond_t conditionVariable;
  int releaseCount;
  int count;
};

#endif

#ifdef __cplusplus
extern "C" {
#endif

// Create thread.
CUTThread cutStartThread(CUT_THREADROUTINE, void *data);

// Wait for thread to finish.
void cutEndThread(CUTThread thread);

// Destroy thread.
void cutDestroyThread(CUTThread thread);

// Wait for multiple threads.
void cutWaitForThreads(const CUTThread *threads, int num);

// Create barrier.
CUTBarrier cutCreateBarrier(int releaseCount);

// Increment barrier. (excution continues)
void cutIncrementBarrier(CUTBarrier *barrier);

// Wait for barrier release.
void cutWaitForBarrier(CUTBarrier *barrier);

// Destory barrier
void cutDestroyBarrier(CUTBarrier *barrier);

#ifdef __cplusplus
}  // extern "C"
#endif

#if _WIN32
// Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data) {
  return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
}

// Wait for thread to finish
void cutEndThread(CUTThread thread) {
  WaitForSingleObject(thread, INFINITE);
  CloseHandle(thread);
}

// Destroy thread
void cutDestroyThread(CUTThread thread) {
  TerminateThread(thread, 0);
  CloseHandle(thread);
}

// Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num) {
  WaitForMultipleObjects(num, threads, true, INFINITE);

  for (int i = 0; i < num; i++) {
    CloseHandle(threads[i]);
  }
}

// Create barrier.
CUTBarrier cutCreateBarrier(int releaseCount) {
  CUTBarrier barrier;

  InitializeCriticalSection(&barrier.criticalSection);
  barrier.barrierEvent = CreateEvent(NULL, TRUE, FALSE, TEXT("BarrierEvent"));
  barrier.count = 0;
  barrier.releaseCount = releaseCount;

  return barrier;
}

// Increment barrier. (excution continues)
void cutIncrementBarrier(CUTBarrier *barrier) {
  int myBarrierCount;
  EnterCriticalSection(&barrier->criticalSection);
  myBarrierCount = ++barrier->count;
  LeaveCriticalSection(&barrier->criticalSection);

  if (myBarrierCount >= barrier->releaseCount) {
    SetEvent(barrier->barrierEvent);
  }
}

// Wait for barrier release.
void cutWaitForBarrier(CUTBarrier *barrier) {
  WaitForSingleObject(barrier->barrierEvent, INFINITE);
}

// Destory barrier
void cutDestroyBarrier(CUTBarrier *barrier) {}

#else
// Create thread
inline CUTThread cutStartThread(CUT_THREADROUTINE func, void *data) {
  pthread_t thread;
  pthread_create(&thread, NULL, func, data);
  return thread;
}

// Wait for thread to finish
inline void cutEndThread(CUTThread thread) { pthread_join(thread, NULL); }

// Destroy thread
inline void cutDestroyThread(CUTThread thread) { pthread_cancel(thread); }

// Wait for multiple threads
inline void cutWaitForThreads(const CUTThread *threads, int num) {
  for (int i = 0; i < num; i++) {
    cutEndThread(threads[i]);
  }
}

// Create barrier.
inline CUTBarrier cutCreateBarrier(int releaseCount) {
  CUTBarrier barrier;

  barrier.count = 0;
  barrier.releaseCount = releaseCount;

  pthread_mutex_init(&barrier.mutex, 0);
  pthread_cond_init(&barrier.conditionVariable, 0);

  return barrier;
}

// Increment barrier. (excution continues)
inline void cutIncrementBarrier(CUTBarrier *barrier) {
  int myBarrierCount;
  pthread_mutex_lock(&barrier->mutex);
  myBarrierCount = ++barrier->count;
  pthread_mutex_unlock(&barrier->mutex);

  if (myBarrierCount >= barrier->releaseCount) {
    pthread_cond_signal(&barrier->conditionVariable);
  }
}

// Wait for barrier release.
inline void cutWaitForBarrier(CUTBarrier *barrier) {
  pthread_mutex_lock(&barrier->mutex);

  while (barrier->count < barrier->releaseCount) {
    pthread_cond_wait(&barrier->conditionVariable, &barrier->mutex);
  }

  pthread_mutex_unlock(&barrier->mutex);
}

// Destory barrier
inline void cutDestroyBarrier(CUTBarrier *barrier) {
  pthread_mutex_destroy(&barrier->mutex);
  pthread_cond_destroy(&barrier->conditionVariable);
}

#endif
#endif  // MULTITHREADING_H
