/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2019 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "xpmem_impl.h"
#include "xpmem_noinline.h"
#include "build_nodemap.h"
#include "mpidu_init_shm.h"
#include "xpmem_seg.h"

int MPIDI_XPMEM_mpi_init_hook(int rank, int size, int *n_vcis_provided, int *tag_bits)
{
    int mpi_errno = MPI_SUCCESS;
    int i, my_local_rank = -1, num_local = 0;
    MPIDU_shm_seg_t shm_seg;
    MPIDU_shm_barrier_t *shm_seg_barrier = NULL;
    xpmem_segid_t *xpmem_segids = NULL;
    MPIDI_XPMEM_seg_t *seg_ptr = NULL;
    OPA_int_t *coop_counter = NULL;
    uint64_t *coop_caddr;
    int local_rank_0 = -1;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_XPMEM_INIT_HOOK);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_XPMEM_INIT_HOOK);
    MPIR_CHKPMEM_DECL(1);

#ifdef MPL_USE_DBG_LOGGING
    MPIDI_CH4_SHM_XPMEM_GENERAL = MPL_dbg_class_alloc("SHM_XPMEM", "shm_xpmem");
#endif /* MPL_USE_DBG_LOGGING */

    /* Try to share entire address space */
    MPIDI_XPMEM_global.segid = xpmem_make(0, XPMEM_MAXADDR_SIZE, XPMEM_PERMIT_MODE,
                                          MPIDI_XPMEM_PERMIT_VALUE);
    /* 64-bit segment ID or failure(-1) */
    MPIR_ERR_CHKANDJUMP(MPIDI_XPMEM_global.segid == -1, mpi_errno, MPI_ERR_OTHER, "**xpmem_make");
    XPMEM_TRACE("init: make segid: 0x%lx\n", (uint64_t) MPIDI_XPMEM_global.segid);

    MPIR_NODEMAP_get_local_info(rank, size, MPIDI_global.node_map[0], &num_local,
                                &my_local_rank, &local_rank_0);
    MPIDI_XPMEM_global.num_local = num_local;
    MPIDI_XPMEM_global.local_rank = my_local_rank;
    MPIDI_XPMEM_global.node_group_ptr = NULL;

    MPIDU_Init_shm_put(&MPIDI_XPMEM_global.segid, sizeof(xpmem_segid_t));
    MPIDU_Init_shm_barrier();

    mpi_errno = MPIDU_shm_seg_alloc(sizeof(uint64_t), (void **) &coop_caddr, MPL_MEM_SHM);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIDU_shm_seg_commit(&shm_seg,
                                     &shm_seg_barrier,
                                     num_local, my_local_rank, local_rank_0, rank, MPL_MEM_SHM);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* Only rank 0 need to actually allocate an array to store counters and
     * other processes should attach to this array through MPIDI_XPMEM_seg_regist */
    if (!my_local_rank) {
        coop_counter =
            (OPA_int_t *) MPL_malloc(num_local * num_local * sizeof(OPA_int_t), MPL_MEM_OTHER);
        MPIR_ERR_CHKANDJUMP(coop_counter == NULL, mpi_errno, MPI_ERR_OTHER, "**nomem");

        for (i = 0; i < num_local * num_local; ++i)
            OPA_store_int(&coop_counter[i], 0);

        *coop_caddr = (uint64_t) coop_counter;
    }

    mpi_errno = MPIDU_shm_barrier(shm_seg_barrier, num_local);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* Initialize segmap for every local processes */
    MPIDI_XPMEM_global.segmaps = NULL;
    MPIR_CHKPMEM_MALLOC(MPIDI_XPMEM_global.segmaps, MPIDI_XPMEM_segmap_t *,
                        sizeof(MPIDI_XPMEM_segmap_t) * num_local,
                        mpi_errno, "xpmem segmaps", MPL_MEM_SHM);
    for (i = 0; i < num_local; i++) {
        MPIDU_Init_shm_get(i, sizeof(xpmem_segid_t), &MPIDI_XPMEM_global.segmaps[i].remote_segid);
        MPIDI_XPMEM_global.segmaps[i].apid = -1;        /* get apid at the first communication  */

        /* Init AVL tree based segment cache */
        MPIDI_XPMEM_segtree_init(&MPIDI_XPMEM_global.segmaps[i].segcache);
    }

    /* Initialize other global parameters */
    MPIDI_XPMEM_global.sys_page_sz = (size_t) sysconf(_SC_PAGESIZE);

    /* Attach to local root coop_counter array */
    if (my_local_rank) {
        mpi_errno =
            MPIDI_XPMEM_seg_regist(0, num_local * num_local * sizeof(OPA_int_t),
                                   (void *) *coop_caddr, &seg_ptr, (void **) &coop_counter);
        if (mpi_errno != MPI_SUCCESS)
            MPIR_ERR_POP(mpi_errno);
    }

    /* Initialize other global parameters */
    MPIDI_XPMEM_global.coop_counter = coop_counter;
    MPIDI_XPMEM_global.seg_ptr = seg_ptr;
    MPIDI_XPMEM_global.dmessage_queue = NULL;

    mpi_errno = MPIDU_shm_seg_destroy(&shm_seg, num_local);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_XPMEM_INIT_HOOK);
    return mpi_errno;
  fn_fail:
    MPIR_CHKPMEM_REAP();
    goto fn_exit;
}

int MPIDI_XPMEM_mpi_finalize_hook(void)
{
    int mpi_errno = MPI_SUCCESS;
    int i, ret = 0;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_XPMEM_FINALIZE_HOOK);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_XPMEM_FINALIZE_HOOK);

    /* Free temporary shared buffer */
    if (MPIDI_XPMEM_global.local_rank) {
        mpi_errno = MPIDI_XPMEM_seg_deregist(MPIDI_XPMEM_global.seg_ptr);
        if (mpi_errno != MPI_SUCCESS)
            MPIR_ERR_POP(mpi_errno);
    } else {
        MPL_free(MPIDI_XPMEM_global.coop_counter);
    }

    for (i = 0; i < MPIDI_XPMEM_global.num_local; i++) {
        /* should be called before xpmem_release
         * MPIDI_XPMEM_segtree_tree_delete_all will call xpmem_detach */
        MPIDI_XPMEM_segtree_delete_all(&MPIDI_XPMEM_global.segmaps[i].segcache);
        if (MPIDI_XPMEM_global.segmaps[i].apid != -1) {
            XPMEM_TRACE("finalize: release apid: node_rank %d, 0x%lx\n",
                        i, (uint64_t) MPIDI_XPMEM_global.segmaps[i].apid);
            ret = xpmem_release(MPIDI_XPMEM_global.segmaps[i].apid);
            /* success(0) or failure(-1) */
            MPIR_ERR_CHKANDJUMP(ret == -1, mpi_errno, MPI_ERR_OTHER, "**xpmem_release");
        }
    }

    MPL_free(MPIDI_XPMEM_global.segmaps);

    if (MPIDI_XPMEM_global.segid != -1) {
        XPMEM_TRACE("finalize: remove segid: 0x%lx\n", (uint64_t) MPIDI_XPMEM_global.segid);
        ret = xpmem_remove(MPIDI_XPMEM_global.segid);
        /* success(0) or failure(-1) */
        MPIR_ERR_CHKANDJUMP(ret == -1, mpi_errno, MPI_ERR_OTHER, "**xpmem_remove");
    }

    if (MPIDI_XPMEM_global.node_group_ptr) {
        mpi_errno = MPIR_Group_free_impl(MPIDI_XPMEM_global.node_group_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_XPMEM_FINALIZE_HOOK);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
