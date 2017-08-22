// CUDA debugging helpers
#pragma once

#include <stdio.h>


// macOS does not support device side assertions
#ifdef __APPLE__
#define D_ASSERT(predicate) \
    if (!(predicate))       \
    printf("(%s:%d) Device assertion failed!\n", __FILE__, __LINE__)
#else
#define D_ASSERT(predicate) assert(predicate)
#endif


// Wrapper to check for CUDA errors, see https://devblogs.nvidia.com/
// parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
inline void cudaErrorCheck(const char *file, int line)
{
    cudaError_t syncError = cudaGetLastError();
    cudaError_t asyncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("Sync CUDA error: %s, %s(%d).\n", cudaGetErrorString(syncError),
            file, line);
        exit(-1);
    }
    if (asyncError != cudaSuccess) {
        printf("Async CUDA error: %s, %s(%d).\n",
            cudaGetErrorString(asyncError), file, line);
        exit(-1);
    }
}

#define CHECK_CUDA cudaErrorCheck(__FILE__, __LINE__)
