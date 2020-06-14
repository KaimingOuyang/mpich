/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef GPU_PRE_H_INCLUDED
#define GPU_PRE_H_INCLUDED

#include "mpl.h"

typedef MPL_gpu_ipc_mem_handle_t MPIDI_GPU_ipc_handle_t;

typedef struct {
    int **visible_dev_global_id;
    int *local_ranks;
    int *local_procs;
} MPIDI_GPU_global_t;

extern MPIDI_GPU_global_t MPIDI_gpu_global;

#endif /* GPU_PRE_H_INCLUDED */
