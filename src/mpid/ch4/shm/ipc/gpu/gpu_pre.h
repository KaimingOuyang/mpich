/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef GPU_PRE_H_INCLUDED
#define GPU_PRE_H_INCLUDED

#include "uthash.h"

typedef struct MPIDI_GPU_mem_handle {
    MPL_gpu_ipc_mem_handle_t ipc_handle;
    int global_dev_id;
    int src_dt_contig;
} MPIDI_GPU_mem_handle_t;

typedef struct MPIDI_gpu_dev_id {
    int local_dev_id;
    int global_dev_id;
    UT_hash_handle hh;
} MPIDI_gpu_dev_id_t;

typedef struct {
    MPIDI_gpu_dev_id_t *local_to_global_map;
    MPIDI_gpu_dev_id_t *global_to_local_map;
    int **visible_dev_global_id;
    int *local_ranks;
    int *local_procs;
} MPIDI_GPU_global_t;

extern MPIDI_GPU_global_t MPIDI_gpu_global;

#endif /* GPU_PRE_H_INCLUDED */
