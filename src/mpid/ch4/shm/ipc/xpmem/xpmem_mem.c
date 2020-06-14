/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */
#include "xpmem_seg.h"
#include "xpmem_post.h"

int MPIDI_XPMEM_ipc_handle_map(int node_rank,
                               MPIDI_XPMEM_ipc_handle_t handle, size_t size, void **vaddr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_XPMEM_IPC_HANDLE_MAP);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_XPMEM_IPC_HANDLE_MAP);

    mpi_errno = MPIDI_XPMEMI_seg_regist(node_rank, size, (void *) handle,
                                        vaddr,
                                        &MPIDI_XPMEMI_global.segmaps[node_rank].segcache_ubuf);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_XPMEM_IPC_HANDLE_MAP);
    return mpi_errno;
}
