/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef GPU_PRE_H_INCLUDED
#define GPU_PRE_H_INCLUDED

typedef struct MPIDI_GPU_mem_handle {
    MPL_gpu_ipc_mem_handle_t ipc_handle;
} MPIDI_GPU_mem_handle_t;

typedef struct {
    int **visible_dev_global_id;
    int *local_ranks;
    int *local_procs;
} MPIDI_GPU_global_t;

extern MPIDI_GPU_global_t MPIDI_gpu_global;

#endif /* GPU_PRE_H_INCLUDED */
