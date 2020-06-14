/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */
#ifndef GPU_POST_H_INCLUDED
#define GPU_POST_H_INCLUDED

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD
      category    : CH4
      type        : int
      default     : 32768
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        If a send message size is greater than or equal to MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD (in
        bytes), then enable GPU-based single copy protocol for intranode communication. The
        environment variable is valid only when then GPU IPC shmmod is enabled.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

#include "ch4_impl.h"
#include "gpu_pre.h"

MPL_STATIC_INLINE_PREFIX int MPIDI_GPU_get_ipc_attr(const void *vaddr,
                                                    MPIDI_IPCI_ipc_attr_t * ipc_attr)
{
    int mpi_errno = MPI_SUCCESS, mpl_err = MPL_SUCCESS;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_IPC_ATTR_CREATE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_IPC_ATTR_CREATE);

#ifdef MPIDI_CH4_SHM_ENABLE_GPU
    ipc_attr->ipc_type = MPIDI_IPCI_TYPE__GPU;
    mpl_err = MPL_gpu_ipc_handle_create(vaddr, &ipc_attr->ipc_handle.gpu);
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                        "**gpu_ipc_handle_create");

    ipc_attr->threshold.send_lmt_sz = MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD;
#else
    /* Do not support IPC data transfer */
    ipc_attr->ipc_type = MPIDI_IPCI_TYPE__NONE;
    ipc_attr->threshold.send_lmt_sz = MPIR_AINT_MAX;
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_IPC_ATTR_CREATE);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_GPU_ipc_handle_map(int node_rank,
                                                      MPIDI_GPU_ipc_handle_t ipc_handle,
                                                      MPL_gpu_device_handle_t dev_handle,
                                                      int src_dt_contig, MPI_Datatype recv_type,
                                                      void **vaddr)
{
    int remote_dev_id;
    MPL_gpu_device_handle_t remote_dev_handle;
    int mpi_errno = MPI_SUCCESS, mpl_err = MPL_SUCCESS;
    int remote_global_dev_id, recv_dt_contig;
    uintptr_t recv_dt_sz;
    MPI_Aint iov_len;
    double recv_dt_density;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_ATTACH_MEM);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_ATTACH_MEM);

    MPIDI_Datatype_check_contig(recv_type, recv_dt_contig);
    MPL_gpu_ipc_handle_get_global_dev_id(ipc_handle, &remote_global_dev_id);
    if (!MPIDI_gpu_global.visible_dev_global_id[node_rank][remote_global_dev_id]) {
        mpl_err = MPL_gpu_ipc_handle_map(ipc_handle, dev_handle, vaddr);
    } else {
        MPL_gpu_ipc_handle_get_local_dev(ipc_handle, &remote_dev_handle);
        if (src_dt_contig && !recv_dt_contig)
            mpl_err = MPL_gpu_ipc_handle_map(ipc_handle, dev_handle, vaddr);
        else if ((src_dt_contig && recv_dt_contig) || (!src_dt_contig && recv_dt_contig))
            mpl_err = MPL_gpu_ipc_handle_map(ipc_handle, remote_dev_handle, vaddr);
        else {
            /* TODO: after get remote datatype, we can compare remote and local datatype density
             * to decide which gpu to map buffer onto. For now, we just add dummy mapping. */
            MPIR_Datatype_get_density(recv_type, recv_dt_density);
            mpl_err = MPL_gpu_ipc_handle_map(ipc_handle, dev_handle, vaddr);
        }
    }
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**gpu_ipc_handle_map");

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_IPC_HANDLE_MAP);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_GPU_mpi_init_hook(int rank, int size, int *tag_bits);
int MPIDI_GPU_mpi_finalize_hook(void);

#endif /* GPU_POST_H_INCLUDED */
