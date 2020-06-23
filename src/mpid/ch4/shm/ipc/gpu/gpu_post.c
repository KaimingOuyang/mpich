/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpidimpl.h"
#include "gpu_pre.h"
#include "gpu_types.h"

int MPIDI_GPU_ipc_handle_create(const void *vaddr, int rank, MPIR_Comm * comm,
                                MPL_gpu_device_handle_t device, MPIDI_GPU_ipc_handle_t * handle)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_IPC_HANDLE_CREATE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_IPC_HANDLE_CREATE);

#ifdef MPIDI_CH4_SHM_ENABLE_GPU
    int local_dev_id, grank, mpl_err;
    MPIDI_GPUI_dev_id_t *tmp;
    grank = MPIDIU_rank_to_lpid(rank, comm);
    mpl_err =
        MPL_gpu_ipc_handle_create(vaddr, MPIDI_GPUI_global.local_ranks[grank], &handle->ipc_handle);
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                        "**gpu_ipc_handle_create");

    MPL_gpu_get_dev_id(device, &local_dev_id);
    HASH_FIND_INT(MPIDI_GPUI_global.local_to_global_map, &local_dev_id, tmp);
    handle->global_dev_id = tmp->global_dev_id;
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_IPC_HANDLE_CREATE);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_GPU_get_ipc_attr(MPIDI_IPCI_ipc_attr_t * ipc_attr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_GET_IPC_ATTR);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_GET_IPC_ATTR);

#ifdef MPIDI_CH4_SHM_ENABLE_GPU
    ipc_attr->ipc_type = MPIDI_IPCI_TYPE__GPU;
    ipc_attr->threshold.send_lmt_sz = MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD;
#else
    /* Do not support IPC data transfer */
    ipc_attr->ipc_type = MPIDI_IPCI_TYPE__NONE;
    ipc_attr->threshold.send_lmt_sz = MPIR_AINT_MAX;
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_GET_IPC_ATTR);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_GPU_ipc_handle_map(MPIDI_GPU_ipc_handle_t handle,
                             MPL_gpu_device_handle_t dev_handle,
                             MPI_Datatype recv_type, void **vaddr)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_IPC_HANDLE_MAP);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_IPC_HANDLE_MAP);

#ifdef MPIDI_CH4_SHM_ENABLE_GPU
    int recv_dev_id;
    int recv_dt_contig, mpl_err = MPL_SUCCESS;
    MPIDI_GPUI_dev_id_t *avail_id = NULL;
    MPIDI_Datatype_check_contig(recv_type, recv_dt_contig);
    HASH_FIND_INT(MPIDI_GPUI_global.global_to_local_map, &handle.global_dev_id, avail_id);

    MPL_gpu_get_dev_id(dev_handle, &recv_dev_id);
    if (recv_dev_id < 0) {
        /* when receiver's buffer is on host memory, recv_dev_id will be less than 0.
         * however, when we decide to map buffer onto receiver's device, this mapping
         * will be invalid, so we need to assign a default gpu instead; for now, we
         * assume process can at least access one GPU, so device id 0 is set. */
        recv_dev_id = 0;
        MPL_gpu_get_dev_handle(recv_dev_id, &dev_handle);
    }

    if (avail_id == NULL)
        mpl_err = MPL_gpu_ipc_handle_map(handle.ipc_handle, dev_handle, vaddr);
    else {
        MPL_gpu_device_handle_t remote_dev_handle;
        MPL_gpu_get_dev_handle(avail_id->local_dev_id, &remote_dev_handle);
        if (!recv_dt_contig)
            mpl_err = MPL_gpu_ipc_handle_map(handle.ipc_handle, dev_handle, vaddr);
        else
            mpl_err = MPL_gpu_ipc_handle_map(handle.ipc_handle, remote_dev_handle, vaddr);
    }
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**gpu_ipc_handle_map");
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_IPC_HANDLE_MAP);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
