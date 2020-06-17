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

MPL_STATIC_INLINE_PREFIX int MPIDI_GPU_get_mem_attr(const void *vaddr, int dt_contig,
                                                    MPIDI_IPCI_mem_attr_t * attr)
{
    int mpi_errno = MPI_SUCCESS, mpl_err = MPL_SUCCESS;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_GET_MEM_ATTR);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_IPC_GET_MEM_ATTR);

#ifdef MPIDI_CH4_SHM_ENABLE_GPU
    int local_dev_id;
    MPIDI_gpu_dev_id_t *tmp;

    attr->ipc_type = MPIDI_IPCI_TYPE__GPU;
    mpl_err = MPL_gpu_ipc_handle_create(vaddr, &attr->mem_handle.gpu.ipc_handle);
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER,
                        "**gpu_ipc_handle_create");

    MPL_gpu_get_dev_id(attr->gpu_attr.device, &local_dev_id);
    HASH_FIND_INT(MPIDI_gpu_global.local_to_global_map, &local_dev_id, tmp);
    attr->mem_handle.gpu.global_dev_id = tmp->global_dev_id;
    attr->mem_handle.gpu.src_dt_contig = dt_contig;
    attr->threshold.send_lmt_sz = MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD;
#else
    /* Do not support IPC data transfer */
    attr->ipc_type = MPIDI_IPCI_TYPE__NONE;
    attr->threshold.send_lmt_sz = MPIR_AINT_MAX;
#endif

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_GET_MEM_ATTR);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_GPU_attach_mem(MPIDI_GPU_mem_handle_t mem_handle,
                                                  MPL_gpu_device_handle_t dev_handle,
                                                  MPI_Datatype recv_type, void **vaddr)
{
    int mpi_errno = MPI_SUCCESS, mpl_err = MPL_SUCCESS;
    int recv_dt_contig;
    MPIDI_gpu_dev_id_t *avail_id = NULL;
    int src_dt_contig = mem_handle.src_dt_contig;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_ATTACH_MEM);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_ATTACH_MEM);

    MPIDI_Datatype_check_contig(recv_type, recv_dt_contig);
    HASH_FIND_INT(MPIDI_gpu_global.global_to_local_map, &mem_handle.global_dev_id, avail_id);

    if (avail_id == NULL)
        mpl_err = MPL_gpu_ipc_handle_map(mem_handle.ipc_handle, dev_handle, vaddr);
    else {
        MPL_gpu_device_handle_t remote_dev_handle;
        MPL_gpu_get_dev_handle(avail_id->local_dev_id, &remote_dev_handle);
        if (src_dt_contig && !recv_dt_contig)
            mpl_err = MPL_gpu_ipc_handle_map(mem_handle.ipc_handle, dev_handle, vaddr);
        else if (((src_dt_contig && recv_dt_contig) || (!src_dt_contig && recv_dt_contig)))
            mpl_err = MPL_gpu_ipc_handle_map(mem_handle.ipc_handle, remote_dev_handle, vaddr);
        else {
            /* TODO: after get remote datatype, we can compare remote and local datatype density
             * to decide which gpu to map buffer onto. For now, we just add dummy mapping. */
            mpl_err = MPL_gpu_ipc_handle_map(mem_handle.ipc_handle, dev_handle, vaddr);
        }
    }
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**gpu_ipc_handle_map");

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_ATTACH_MEM);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_GPU_close_mem(void *vaddr, MPIDI_GPU_mem_handle_t handle)
{
    int mpi_errno = MPI_SUCCESS, mpl_err = MPL_SUCCESS;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_GPU_CLOSE_MEM);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_GPU_CLOSE_MEM);

    mpl_err = MPL_gpu_ipc_handle_unmap(vaddr, handle.ipc_handle);
    MPIR_ERR_CHKANDJUMP(mpl_err != MPL_SUCCESS, mpi_errno, MPI_ERR_OTHER, "**gpu_ipc_handle_unmap");

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_GPU_CLOSE_MEM);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_GPU_mpi_init_hook(int rank, int size, int *tag_bits);
int MPIDI_GPU_mpi_finalize_hook(void);

#endif /* GPU_POST_H_INCLUDED */
