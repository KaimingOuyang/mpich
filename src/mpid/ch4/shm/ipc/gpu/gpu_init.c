/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpidimpl.h"
#include "gpu_post.h"
#include "gpu_pre.h"

int MPIDI_GPU_mpi_init_hook(int rank, int size, int *tag_bits)
{
    int mpl_err, mpi_errno = MPI_SUCCESS;
    int local_max_dev_id, global_max_dev_id = -1;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_MPI_INIT_HOOK);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_MPI_INIT_HOOK);
    MPIR_CHKPMEM_DECL(1);
    mpl_err = MPL_gpu_init(&local_max_dev_id);
    if (mpl_err != MPL_SUCCESS)
        mpi_errno = MPI_ERR_OTHER;

    MPIDU_Init_shm_put(&local_max_dev_id, sizeof(int));
    MPIDU_Init_shm_barrier();

    /* get global max device id */
    for (int i = 0; i < MPIR_Process.local_size; i++) {
        MPIDU_Init_shm_get(i, sizeof(int), &local_max_dev_id);
        if (local_max_dev_id > global_max_dev_id)
            global_max_dev_id = local_max_dev_id;
    }
    MPIDU_Init_shm_barrier();

    MPIR_CHKPMEM_MALLOC(MPIDI_gpu_global.visible_dev_global_id, int **,
                        sizeof(int *) * MPIR_Process.local_size, mpi_errno, "gpu devmaps",
                        MPL_MEM_SHM);
    for (int i = 0; i < MPIR_Process.local_size; ++i) {
        MPIDI_gpu_global.visible_dev_global_id[i] =
            (int *) MPL_malloc(sizeof(int) * global_max_dev_id, MPL_MEM_OTHER);
        MPIR_Assert(MPIDI_gpu_global.visible_dev_global_id[i]);

        if (i == MPIR_Process.local_rank) {
            MPL_gpu_get_global_visiable_dev(MPIDI_gpu_global.visible_dev_global_id[i],
                                            global_max_dev_id);
            MPIDU_Init_shm_put(MPIDI_gpu_global.visible_dev_global_id[i],
                               sizeof(int) * global_max_dev_id);
        }
    }
    MPIDU_Init_shm_barrier();

    /* FIXME: current implementation uses MPIDU_Init_shm_get to exchange visible id.
     * shm buffer size is defined as 64 bytes by default. Therefore, if number of
     * gpu device is larger than 16, the MPIDU_Init_shm_get would fail. */
    for (int i = 0; i < MPIR_Process.local_size; ++i)
        MPIDU_Init_shm_get(i, sizeof(int) * global_max_dev_id,
                           MPIDI_gpu_global.visible_dev_global_id[i]);
    MPIDU_Init_shm_barrier();

    MPIDI_gpu_global.local_procs = MPIR_Process.node_local_map;
    MPIDI_gpu_global.local_ranks = (int *) MPL_malloc(MPIR_Process.size * sizeof(int), MPL_MEM_SHM);
    for (int i = 0; i < MPIR_Process.size; ++i) {
        MPIDI_gpu_global.local_ranks[i] = -1;
    }
    for (int i = 0; i < MPIR_Process.local_size; i++) {
        MPIDI_gpu_global.local_ranks[MPIDI_gpu_global.local_procs[i]] = i;
    }

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_XPMEM_MPI_INIT_HOOK);
    return mpi_errno;
  fn_fail:
    MPIR_CHKPMEM_REAP();
    goto fn_exit;
}

int MPIDI_GPU_mpi_finalize_hook(void)
{
    int mpl_err, mpi_errno = MPI_SUCCESS;

    MPL_free(MPIDI_gpu_global.local_ranks);
    for (int i = 0; i < MPIR_Process.local_size; ++i)
        MPL_free(MPIDI_gpu_global.visible_dev_global_id[i]);
    MPL_free(MPIDI_gpu_global.visible_dev_global_id);

    mpl_err = MPL_gpu_finalize();
    if (mpl_err != MPL_SUCCESS)
        mpi_errno = MPI_ERR_OTHER;

    return mpi_errno;
}
