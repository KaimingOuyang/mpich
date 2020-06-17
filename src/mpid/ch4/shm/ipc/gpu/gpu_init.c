/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpidimpl.h"
#include "gpu_post.h"

int MPIDI_GPU_mpi_init_hook(int rank, int size, int *tag_bits)
{
    int mpl_err, mpi_errno = MPI_SUCCESS;
    int device_count;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_MPI_INIT_HOOK);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_MPI_INIT_HOOK);

    mpl_err = MPL_gpu_init(&device_count);
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**gpu_init");

    int *global_ids = MPL_malloc(sizeof(int) * device_count, MPL_MEM_OTHER);
    assert(global_ids);

    mpl_err = MPL_gpu_get_global_dev_ids(global_ids, device_count);
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                        "**gpu_get_global_dev_ids");

    MPIDI_gpu_global.local_to_global_map = NULL;
    MPIDI_gpu_global.global_to_local_map = NULL;
    for (int i = 0; i < device_count; ++i) {
        MPIDI_gpu_dev_id_t *id_obj =
            (MPIDI_gpu_dev_id_t *) MPL_malloc(sizeof(MPIDI_gpu_dev_id_t), MPL_MEM_OTHER);
        assert(id_obj);
        id_obj->local_dev_id = i;
        id_obj->global_dev_id = global_ids[i];
        HASH_ADD_INT(MPIDI_gpu_global.local_to_global_map, local_dev_id, id_obj, MPL_MEM_OTHER);

        id_obj = (MPIDI_gpu_dev_id_t *) MPL_malloc(sizeof(MPIDI_gpu_dev_id_t), MPL_MEM_OTHER);
        assert(id_obj);
        id_obj->local_dev_id = i;
        id_obj->global_dev_id = global_ids[i];
        HASH_ADD_INT(MPIDI_gpu_global.global_to_local_map, global_dev_id, id_obj, MPL_MEM_OTHER);
    }

    MPL_free(global_ids);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_MPI_INIT_HOOK);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_GPU_mpi_finalize_hook(void)
{
    int mpl_err, mpi_errno = MPI_SUCCESS;
    MPIDI_gpu_dev_id_t *current, *tmp;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_MPI_FINALIZE_HOOK);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_MPI_FINALIZE_HOOK);

    HASH_ITER(hh, MPIDI_gpu_global.local_to_global_map, current, tmp) {
        HASH_DEL(MPIDI_gpu_global.local_to_global_map, current);
        MPL_free(current);
    }

    HASH_ITER(hh, MPIDI_gpu_global.global_to_local_map, current, tmp) {
        HASH_DEL(MPIDI_gpu_global.global_to_local_map, current);
        MPL_free(current);
    }

    mpl_err = MPL_gpu_finalize();
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**gpu_finalize");

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_MPI_FINALIZE_HOOK);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
