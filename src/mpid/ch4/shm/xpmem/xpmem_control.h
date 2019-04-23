/*
 *  (C) 2006 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 *  Portions of this code were written by Intel Corporation.
 *  Copyright (C) 2011-2017 Intel Corporation.  Intel provides this material
 *  to Argonne National Laboratory subject to Software Grant and Corporate
 *  Contributor License Agreement dated February 8, 2012.
 */
#ifndef XPMEM_CONTROL_H_INCLUDED
#define XPMEM_CONTROL_H_INCLUDED

#include "ch4_impl.h"
#include "xpmem_pre.h"
#include "xpmem_impl.h"
#include "xpmem_recv.h"
#include "xpmem_send.h"
#include "datatype.h"


#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_ctrl_do_send_lmt_req_cb
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_do_send_lmt_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr)
{
    int mpi_errno = MPI_SUCCESS, recvtype_iscontig;
    MPIDI_SHM_ctrl_xpmem_send_lmt_req_t *slmt_req_hdr = &ctrl_hdr->xpmem_slmt_req;
    MPIR_Request *rreq = NULL;
    MPIR_Comm *root_comm;
    MPIDI_av_entry_t *av;
    MPIR_Request *anysource_partner;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_CTRL_DO_SEND_LMT_REQ_CB);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_CTRL_DO_SEND_LMT_REQ_CB);

    XPMEM_PT2PT_DBG_PRINT
        ("send_lmt_req_cb: received src_offset 0x%lx, data_sz 0x%lx, sreq_ptr 0x%lx, "
         "src_lrank %d, match info[src_rank %d, tag %d, context_id 0x%x]\n",
         slmt_req_hdr->src_offset, slmt_req_hdr->data_sz, slmt_req_hdr->sreq_ptr,
         slmt_req_hdr->src_lrank, slmt_req_hdr->src_rank, slmt_req_hdr->tag,
         slmt_req_hdr->context_id);

    /* Try to match a posted receive request.
     * root_comm cannot be NULL if a posted receive request exists, because
     * we increase its refcount at enqueue time. */
    root_comm = MPIDIG_context_id_to_comm(slmt_req_hdr->context_id);
    if (root_comm) {
        int continue_matching = 1;
        while (continue_matching) {
            anysource_partner = NULL;

            rreq = MPIDIG_dequeue_posted(slmt_req_hdr->src_rank, slmt_req_hdr->tag,
                                         slmt_req_hdr->context_id,
                                         &MPIDIG_COMM(root_comm, posted_list));

            if (rreq && MPIDI_REQUEST_ANYSOURCE_PARTNER(rreq)) {
                /* Try to cancel NM parter request */
                anysource_partner = MPIDI_REQUEST_ANYSOURCE_PARTNER(rreq);
                mpi_errno = MPIDI_anysource_matched(anysource_partner,
                                                    MPIDI_SHM, &continue_matching);
                if (MPI_SUCCESS != mpi_errno)
                    MPIR_ERR_POP(mpi_errno);

                if (continue_matching) {
                    /* NM partner request has already been matched, we need to continue until
                     * no matching rreq. This SHM rreq will be cancelled by NM. */
                    MPIR_Comm_release(root_comm);       /* -1 for posted_list */
                    MPIR_Datatype_release_if_not_builtin(MPIDIG_REQUEST(rreq, datatype));
                    continue;
                }

                /* Release cancelled NM partner request (only SHM request is returned to user) */
                MPIDI_REQUEST_ANYSOURCE_PARTNER(rreq) = NULL;
                MPIDI_REQUEST_ANYSOURCE_PARTNER(anysource_partner) = NULL;
                MPIR_Request_free(anysource_partner);
            }
            break;
        }
    }

    if (rreq) {
        /* Matching receive was posted */
        MPIDIG_REQUEST(rreq, rank) = slmt_req_hdr->src_rank;
        MPIDIG_REQUEST(rreq, tag) = slmt_req_hdr->tag;
        MPIDIG_REQUEST(rreq, context_id) = slmt_req_hdr->context_id;

        /* Complete XPMEM receive */
#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
        /* TODO: need to support non-contig datatype on receiver side. */
        MPIR_Datatype_iscontig(MPIDIG_REQUEST(rreq, datatype), &recvtype_iscontig);
        if (recvtype_iscontig) {
            av = MPIDIU_comm_rank_to_av(root_comm, slmt_req_hdr->src_rank);
            MPIDI_XPMEM_REQUEST(rreq, sreq_ptr) = slmt_req_hdr->sreq_ptr;

            mpi_errno = MPIDI_XPMEM_lmt_coop_isend(MPIDIG_REQUEST(rreq, buffer),
                                                   MPIDIG_REQUEST(rreq, count),
                                                   MPIDIG_REQUEST(rreq, datatype),
                                                   slmt_req_hdr->src_rank, slmt_req_hdr->tag,
                                                   root_comm, MPIR_CONTEXT_INTRA_PT2PT, av,
                                                   MPIDI_SHM_XPMEM_SEND_LMT_CTS, &rreq);
            if (MPI_SUCCESS != mpi_errno)
                MPIR_ERR_POP(mpi_errno);
        }

        mpi_errno =
            MPIDI_XPMEM_handle_lmt_coop_recv(slmt_req_hdr->src_offset, slmt_req_hdr->data_sz,
                                             slmt_req_hdr->sreq_ptr, slmt_req_hdr->src_lrank,
                                             slmt_req_hdr->call_type, MPIDI_SHM_XPMEM_SEND_LMT_RTS,
                                             rreq);
        if (MPI_SUCCESS != mpi_errno)
            MPIR_ERR_POP(mpi_errno);
#else
        mpi_errno = MPIDI_XPMEM_handle_lmt_recv(slmt_req_hdr->src_offset, slmt_req_hdr->data_sz,
                                                slmt_req_hdr->sreq_ptr, slmt_req_hdr->src_lrank,
                                                rreq);
#endif

        MPIR_Comm_release(root_comm);   /* -1 for posted_list */

        if (MPI_SUCCESS != mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    } else {
        /* Enqueue unexpected receive request */
        rreq = MPIDIG_request_create(MPIR_REQUEST_KIND__RECV, 2);
        MPIR_ERR_CHKANDSTMT(rreq == NULL, mpi_errno, MPIX_ERR_NOREQ, goto fn_fail, "**nomemreq");

        /* store CH4 am rreq info */
        MPIDIG_REQUEST(rreq, buffer) = NULL;
        MPIDIG_REQUEST(rreq, datatype) = MPI_BYTE;
        MPIDIG_REQUEST(rreq, count) = slmt_req_hdr->data_sz;
        MPIDIG_REQUEST(rreq, rank) = slmt_req_hdr->src_rank;
        MPIDIG_REQUEST(rreq, tag) = slmt_req_hdr->tag;
        MPIDIG_REQUEST(rreq, context_id) = slmt_req_hdr->context_id;
        MPIDI_REQUEST(rreq, is_local) = 1;

        /* store XPMEM internal info */
        MPIDI_XPMEM_REQUEST(rreq, unexp_rreq).src_offset = slmt_req_hdr->src_offset;
        MPIDI_XPMEM_REQUEST(rreq, unexp_rreq).data_sz = slmt_req_hdr->data_sz;
#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
        MPIDI_XPMEM_REQUEST(rreq, sreq_ptr) = slmt_req_hdr->sreq_ptr;
        MPIDI_XPMEM_REQUEST(rreq, unexp_rreq).call_type = slmt_req_hdr->call_type;
#else
        MPIDI_XPMEM_REQUEST(rreq, unexp_rreq).sreq_ptr = slmt_req_hdr->sreq_ptr;
#endif
        MPIDI_XPMEM_REQUEST(rreq, unexp_rreq).src_lrank = slmt_req_hdr->src_lrank;
        MPIDI_SHM_REQUEST(rreq, status) |= MPIDI_SHM_REQ_XPMEM_SEND_LMT;

        if (root_comm) {
            MPIR_Comm_add_ref(root_comm);       /* +1 for unexp_list */
            MPIDIG_enqueue_unexp(rreq, &MPIDIG_COMM(root_comm, unexp_list));
        } else {
            MPIDIG_enqueue_unexp(rreq,
                                 MPIDIG_context_id_to_uelist(MPIDIG_REQUEST(rreq, context_id)));
        }

        XPMEM_PT2PT_DBG_PRINT("send_lmt_req_cb: enqueue unexpected, rreq=%p\n", rreq);
    }

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_CTRL_DO_SEND_LMT_REQ_CB);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_ctrl_send_lmt_ack_cb
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_send_lmt_ack_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Request *sreq = (MPIR_Request *) ctrl_hdr->xpmem_slmt_ack.sreq_ptr;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_CTRL_SEND_LMT_ACK_CB);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_CTRL_SEND_LMT_ACK_CB);

    XPMEM_PT2PT_DBG_PRINT("send_lmt_ack_cb: complete sreq %p\n", sreq);
    MPIR_Datatype_release_if_not_builtin(MPIDIG_REQUEST(sreq, datatype));
    MPID_Request_complete(sreq);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_CTRL_SEND_LMT_ACK_CB);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_ctrl_recv_lmt_ack_cb
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_recv_lmt_ack_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Request *rreq = (MPIR_Request *) ctrl_hdr->xpmem_slmt_ack.sreq_ptr;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_CTRL_RECV_LMT_ACK_CB);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_CTRL_RECV_LMT_ACK_CB);

    XPMEM_PT2PT_DBG_PRINT("recv_lmt_ack_cb: complete rreq %p\n", rreq);

    MPIR_Datatype_release_if_not_builtin(MPIDIG_REQUEST(rreq, datatype));
    MPID_Request_complete(rreq);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_CTRL_RECV_LMT_ACK_CB);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_ctrl_send_lmt_rts_req_cb
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_send_lmt_rts_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_SHM_ctrl_xpmem_send_lmt_req_t *slmt_req_hdr =
        &((MPIDI_SHM_ctrl_hdr_t *) ctrl_hdr)->xpmem_slmt_req;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_XPMEM_CTRL_SEND_LMT_RTS_REQ_CB);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_XPMEM_CTRL_SEND_LMT_RTS_REQ_CB);

    mpi_errno = MPIDI_XPMEM_ctrl_do_send_lmt_req_cb(ctrl_hdr);
    if (MPI_SUCCESS != mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_ctrl_send_lmt_cts_req_cb
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_send_lmt_cts_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Request *sreq;
    MPIDI_SHM_ctrl_xpmem_send_lmt_req_t *slmt_req_hdr = &ctrl_hdr->xpmem_slmt_req;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_XPMEM_CTRL_SEND_LMT_CTS_REQ_CB);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_XPMEM_CTRL_SEND_LMT_CTS_REQ_CB);

    /* Sender gets the CTS packet from receiver and need to perform copy */
    sreq = (MPIR_Request *) slmt_req_hdr->sreq_ptr;
    MPIDIG_REQUEST(sreq, rank) = slmt_req_hdr->src_rank;
    MPIDIG_REQUEST(sreq, tag) = slmt_req_hdr->tag;
    MPIDIG_REQUEST(sreq, context_id) = slmt_req_hdr->context_id;

    /* Complete XPMEM receive */
    mpi_errno =
        MPIDI_XPMEM_handle_lmt_coop_recv(slmt_req_hdr->src_offset, slmt_req_hdr->data_sz,
                                         slmt_req_hdr->rreq_ptr, slmt_req_hdr->src_lrank,
                                         slmt_req_hdr->call_type,
                                         MPIDI_SHM_XPMEM_SEND_LMT_CTS, sreq);
    if (MPI_SUCCESS != mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_XPMEM_CTRL_SEND_LMT_CTS_REQ_CB);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif /* MPIDI_CH4_SHM_XPMEM_COOP_P2P */

#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_ctrl_send_lmt_req_cb
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_ctrl_send_lmt_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_CTRL_SEND_LMT_REQ_CB);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_CTRL_SEND_LMT_REQ_CB);

    mpi_errno = MPIDI_XPMEM_ctrl_do_send_lmt_req_cb(ctrl_hdr);
    if (MPI_SUCCESS != mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_CTRL_SEND_LMT_REQ_CB);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif /* XPMEM_CONTROL_H_INCLUDED */
