/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2018 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written by Intel Corporation.
 *  Copyright (C) 2011-2018 Intel Corporation.  Intel provides this material
 *  to Argonne National Laboratory subject to Software Grant and Corporate
 *  Contributor License Agreement dated February 8, 2012.
 */

#ifndef SHM_CONTROL_H_INCLUDED
#define SHM_CONTROL_H_INCLUDED

#include "shm_types.h"
#include "../xpmem/xpmem_pre.h"
#include "../posix/posix_am.h"

#ifdef MPIDI_CH4_SHM_ENABLE_XPMEM
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_send_lmt_ack_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);
#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_recv_lmt_ack_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_send_lmt_rts_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_send_lmt_cts_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);
#else
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_send_lmt_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);
#endif
#endif

#undef FUNCNAME
#define FUNCNAME MPIDI_SHM_ctrl_dispatch
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_ctrl_dispatch(int ctrl_id, void *ctrl_hdr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_CTRL_DISPATCH);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_CTRL_DISPATCH);

    switch (ctrl_id) {
#ifdef MPIDI_CH4_SHM_ENABLE_XPMEM
#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
        case MPIDI_SHM_XPMEM_SEND_LMT_RTS:
            MPIDI_XPMEM_ctrl_send_lmt_rts_req_cb((MPIDI_SHM_ctrl_hdr_t *) ctrl_hdr);
            break;
        case MPIDI_SHM_XPMEM_SEND_LMT_CTS:
            MPIDI_XPMEM_ctrl_send_lmt_cts_req_cb((MPIDI_SHM_ctrl_hdr_t *) ctrl_hdr);
            break;
#else
        case MPIDI_SHM_XPMEM_SEND_LMT_REQ:
            MPIDI_XPMEM_ctrl_send_lmt_req_cb((MPIDI_SHM_ctrl_hdr_t *) ctrl_hdr);
            break;
#endif
        case MPIDI_SHM_XPMEM_SEND_LMT_ACK:
            MPIDI_XPMEM_ctrl_send_lmt_ack_cb((MPIDI_SHM_ctrl_hdr_t *) ctrl_hdr);
            break;
#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
        case MPIDI_SHM_XPMEM_RECV_LMT_ACK:
            MPIDI_XPMEM_ctrl_recv_lmt_ack_cb((MPIDI_SHM_ctrl_hdr_t *) ctrl_hdr);
            break;
#endif

#endif
        default:
            /* Unknown SHM control header */
            MPIR_Assert(0);
    }

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_CTRL_DISPATCH);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_SHM_do_ctrl_send
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_do_ctrl_send(int rank, MPIR_Comm * comm,
                                                    int ctrl_id, void *ctrl_hdr)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_DO_CTRL_SEND);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_DO_CTRL_SEND);

    ret = MPIDI_POSIX_am_send_hdr(rank, comm, MPIDI_POSIX_AM_HDR_SHM,
                                  ctrl_id, ctrl_hdr, sizeof(MPIDI_SHM_ctrl_hdr_t));

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_DO_CTRL_SEND);
    return ret;
}
#endif /* SHM_CONTROL_H_INCLUDED */
