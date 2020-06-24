/*
 *  Copyright (C) by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef MPL_GPU_CUDA_H_INCLUDED
#define MPL_GPU_CUDA_H_INCLUDED

#include "cuda.h"
#include "cuda_runtime_api.h"

typedef struct {
    uintptr_t remote_base_addr;
    uintptr_t len;
    int node_rank;
    cudaIpcMemHandle_t handle;
    uintptr_t offset;
    int handle_status;
} MPL_gpu_ipc_mem_handle_t;
typedef int MPL_gpu_device_handle_t;
#define MPL_GPU_DEVICE_INVALID -1

#endif /* ifndef MPL_GPU_CUDA_H_INCLUDED */
