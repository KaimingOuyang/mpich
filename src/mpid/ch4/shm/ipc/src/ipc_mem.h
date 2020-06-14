/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */
#ifndef IPC_MEM_H_INCLUDED
#define IPC_MEM_H_INCLUDED

#include "ch4_impl.h"
#include "ipc_pre.h"
#include "ipc_types.h"
#include "../xpmem/xpmem_post.h"
#include "../gpu/gpu_post.h"

/* Get local memory handle. No-op if the IPC type is NONE. */
MPL_STATIC_INLINE_PREFIX int MPIDI_IPCI_get_ipc_attr(const void *vaddr,
                                                     MPIDI_IPCI_ipc_attr_t * ipc_attr)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_IPCI_IPC_ATTR_CREATE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_IPCI_IPC_ATTR_CREATE);

    MPIR_GPU_query_pointer_attr(vaddr, &ipc_attr->gpu_attr);

    if (ipc_attr->gpu_attr.type == MPL_GPU_POINTER_DEV) {
        mpi_errno = MPIDI_GPU_get_ipc_attr(vaddr, ipc_attr);
        MPIR_ERR_CHECK(mpi_errno);
    } else {
        mpi_errno = MPIDI_XPMEM_get_ipc_attr(vaddr, ipc_attr);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_IPCI_IPC_ATTR_CREATE);
    return mpi_errno;
  fn_fail:
    ipc_attr->ipc_type = MPIDI_IPCI_TYPE__NONE;
    memset(&ipc_attr->ipc_handle, 0, sizeof(MPIDI_IPCI_ipc_handle_t));
    goto fn_exit;
}

/* Attach remote memory handle. Return the local memory segment handle and
 * the mapped virtual address. No-op if the IPC type is NONE. */
MPL_STATIC_INLINE_PREFIX int MPIDI_IPCI_handle_map(MPIDI_IPCI_type_t ipc_type,
                                                   int node_rank,
                                                   MPIDI_IPCI_ipc_handle_t ipc_handle,
                                                   MPL_gpu_device_handle_t dev_handle, size_t size,
                                                   int src_dt_contig,
                                                   MPI_Datatype recv_type, void **vaddr)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_IPCI_HANDLE_MAP);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_IPCI_HANDLE_MAP);

    switch (ipc_type) {
        case MPIDI_IPCI_TYPE__XPMEM:
            mpi_errno = MPIDI_XPMEM_ipc_handle_map(node_rank, ipc_handle.xpmem, size, vaddr);
            break;
        case MPIDI_IPCI_TYPE__GPU:
            mpi_errno =
                MPIDI_GPU_ipc_handle_map(node_rank, ipc_handle.gpu, dev_handle, src_dt_contig,
                                         recv_type, vaddr);
            break;
        case MPIDI_IPCI_TYPE__NONE:
            /* no-op */
            break;
        default:
            /* Unknown IPC type */
            MPIR_Assert(0);
            break;
    }

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_IPCI_HANDLE_MAP);
    return mpi_errno;
}

#endif /* IPC_MEM_H_INCLUDED */
