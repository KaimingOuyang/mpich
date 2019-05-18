/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2019 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#ifndef XPMEM_RECV_H_INCLUDED
#define XPMEM_RECV_H_INCLUDED

#include "ch4_impl.h"
#include "shm_control.h"
#include "xpmem_noinline.h"
#include "xpmem_control.h"
#include "xpmem_pre.h"
#include "xpmem_seg.h"
#include "xpmem_impl.h"
#include "xpmem_send.h"

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : MPIR_CVAR_CH4_XPMEM_COOP_COPY_CHUNK_SIZE
      category    : CH4
      type        : int
      default     : 32768
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Chunk size is used in XPMEM COOP P2P and determines the concurrency of
        copy and overhead of atomic per chunk copy. Best chunk size should
        balance these two factors.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/


/* Handle and complete a matched XPMEM LMT receive request. Input parameters
 * include send buffer info (see definition in MPIDI_SHM_ctrl_xpmem_send_lmt_req_t)
 * and receive request. */
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_handle_lmt_coop_recv(uint64_t src_offset,
                                                              uint64_t src_data_sz,
                                                              uint64_t req_ptr, int src_lrank,
                                                              int packet_type, MPIR_Request * req)
{
    int mpi_errno = MPI_SUCCESS, dt_contig ATTRIBUTE((unused)), ack_type;
    MPIDI_XPMEM_seg_t *seg_ptr = NULL;
    void *attached_sbuf = NULL;
    void *dest_rbuf = NULL;
    size_t data_sz, recv_data_sz, copy_sz;
    MPI_Aint dt_true_lb;
    MPIR_Datatype *dt_ptr ATTRIBUTE((unused));
    OPA_int_t *counter_ptr;
    int cur_chunk = 0, total_chunk = 0;
    uint64_t cur_offset;

    MPIDI_SHM_ctrl_hdr_t ack_ctrl_hdr;
    MPIDI_SHM_ctrl_xpmem_send_lmt_ack_t *slmt_ack_hdr = &ack_ctrl_hdr.xpmem_slmt_ack;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_HANDLE_LMT_COOP_RECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_HANDLE_LMT_COOP_COOP_RECV);

    mpi_errno = MPIDI_XPMEM_seg_regist(src_lrank, src_data_sz, (void *) src_offset,
                                       &seg_ptr, &attached_sbuf);
    MPIR_ERR_CHECK(mpi_errno);

    MPIDI_Datatype_get_info(MPIDIG_REQUEST(req, count), MPIDIG_REQUEST(req, datatype), dt_contig,
                            data_sz, dt_ptr, dt_true_lb);

    /* Copy data to receive buffer */
    recv_data_sz = MPL_MIN(src_data_sz, data_sz);

    if (packet_type == MPIDI_SHM_XPMEM_SEND_LMT_RTS) {
        /* Receiver get the RTS packet */
        if (src_data_sz > data_sz)
            req->status.MPI_ERROR = MPI_ERR_TRUNCATE;
        /* Set receive status */
        MPIR_STATUS_SET_COUNT(req->status, recv_data_sz);
        req->status.MPI_SOURCE = MPIDIG_REQUEST(req, rank);
        req->status.MPI_TAG = MPIDIG_REQUEST(req, tag);
        counter_ptr = MPIDI_XPMEM_REQUEST(req, counter);
        dest_rbuf =
            (void *) ((char *) MPIDIG_REQUEST(req, buffer) +
                      (counter_ptr == NULL ? 0 : dt_true_lb));
        ack_type = MPIDI_SHM_XPMEM_SEND_LMT_ACK;
    } else {
        /* Sender get the CTS packet */
        dest_rbuf = attached_sbuf;
        attached_sbuf = (void *) ((char *) MPIDIG_REQUEST(req, buffer) + dt_true_lb);
        counter_ptr =
            &MPIDI_XPMEM_global.coop_counter[src_lrank * MPIDI_XPMEM_global.num_local +
                                             MPIDI_XPMEM_global.local_rank];
        ack_type = MPIDI_SHM_XPMEM_RECV_LMT_ACK;
    }

    if (counter_ptr) {
        total_chunk =
            recv_data_sz / MPIR_CVAR_CH4_XPMEM_COOP_COPY_CHUNK_SIZE +
            (recv_data_sz % MPIR_CVAR_CH4_XPMEM_COOP_COPY_CHUNK_SIZE == 0 ? 0 : 1);

        /* TODO: implement OPA_fetch_and_incr_int uint64_t type */
        while ((cur_chunk = OPA_fetch_and_incr_int(counter_ptr)) < total_chunk) {
            cur_offset = ((uint64_t) cur_chunk) * MPIR_CVAR_CH4_XPMEM_COOP_COPY_CHUNK_SIZE;
            copy_sz =
                cur_offset + MPIR_CVAR_CH4_XPMEM_COOP_COPY_CHUNK_SIZE <=
                recv_data_sz ? MPIR_CVAR_CH4_XPMEM_COOP_COPY_CHUNK_SIZE : recv_data_sz - cur_offset;
            mpi_errno =
                MPIR_Localcopy(((char *) attached_sbuf + cur_offset), copy_sz, MPI_BYTE,
                               ((char *) dest_rbuf + cur_offset), copy_sz, MPI_BYTE);
            if (mpi_errno != MPI_SUCCESS)
                MPIR_ERR_POP(mpi_errno);
        }
    } else {
        mpi_errno = MPIR_Localcopy(attached_sbuf, recv_data_sz,
                                   MPI_BYTE, dest_rbuf,
                                   MPIDIG_REQUEST(req, count), MPIDIG_REQUEST(req, datatype));
    }

    XPMEM_TRACE("handle_lmt_recv: handle matched rreq %p [source %d, tag %d, context_id 0x%x],"
                " copy dst %p, src %p, bytes %ld\n", rreq, MPIDIG_REQUEST(rreq, rank),
                MPIDIG_REQUEST(rreq, tag), MPIDIG_REQUEST(rreq, context_id),
                (char *) MPIDIG_REQUEST(rreq, buffer), attached_sbuf, recv_data_sz);

    mpi_errno = MPIDI_XPMEM_seg_deregist(seg_ptr);
    MPIR_ERR_CHECK(mpi_errno);

    /* Send ack */
    if (!counter_ptr || cur_chunk == total_chunk + 1) {
        slmt_ack_hdr->sreq_ptr = req_ptr;
        mpi_errno = MPIDI_SHM_do_ctrl_send(MPIDIG_REQUEST(req, rank),
                                           MPIDIG_context_id_to_comm(MPIDIG_REQUEST
                                                                     (req, context_id)),
                                           ack_type, &ack_ctrl_hdr);
        if (mpi_errno != MPI_SUCCESS)
            MPIR_ERR_POP(mpi_errno);

        if (!counter_ptr) {
            MPIR_Datatype_release_if_not_builtin(MPIDIG_REQUEST(req, datatype));
            MPID_Request_complete(req);
        } else {
            MPIDI_SHM_ctrl_hdr_t ctrl_hdr;
            ctrl_hdr.xpmem_slmt_ack.sreq_ptr = (uint64_t) req;
            mpi_errno = MPIDI_XPMEM_ctrl_do_send_recv_lmt_ack_cb(&ctrl_hdr,
                                                                 packet_type ==
                                                                 MPIDI_SHM_XPMEM_SEND_LMT_RTS ? 0 :
                                                                 1);
        }
    }

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_HANDLE_LMT_COOP_RECV);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_handle_unexp_recv(void *buf,
                                                           MPI_Aint count,
                                                           MPI_Datatype datatype,
                                                           MPIR_Request * message)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_XPMEM_am_unexp_rreq_t *unexp_rreq;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_HANDLE_HANDLE_UNEXP_RECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_HANDLE_HANDLE_UNEXP_RECV);

    unexp_rreq = &MPIDI_XPMEM_REQUEST(message, unexp_rreq);

    MPIR_Comm *root_comm = NULL;
    MPIDI_av_entry_t *av = NULL;
    int recvtype_iscontig;
    MPIR_Datatype_is_contig(datatype, &recvtype_iscontig);
    if (recvtype_iscontig) {
        MPIDI_XPMEM_REQUEST(message, counter) =
            &MPIDI_XPMEM_global.coop_counter[MPIDI_XPMEM_global.local_rank *
                                             MPIDI_XPMEM_global.num_local + unexp_rreq->src_lrank];

        root_comm = MPIDIG_context_id_to_comm(MPIDIG_REQUEST(message, context_id));
        av = MPIDIU_comm_rank_to_av(root_comm, MPIDIG_REQUEST(message, rank));

        mpi_errno = MPIDI_XPMEM_lmt_coop_isend(buf,
                                               count, datatype,
                                               MPIDIG_REQUEST(message, rank),
                                               MPIDIG_REQUEST(message, tag), root_comm,
                                               MPIR_CONTEXT_INTRA_PT2PT, av,
                                               MPIDI_SHM_XPMEM_SEND_LMT_CTS, &message);
        if (MPI_SUCCESS != mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    } else {
        MPIDI_XPMEM_REQUEST(message, counter) = NULL;
    }

    /* Matching XPMEM LMT receive is now posted */
    MPIR_Datatype_add_ref_if_not_builtin(datatype);     /* will -1 once completed in handle_lmt_recv */
    MPIDIG_REQUEST(message, datatype) = datatype;
    MPIDIG_REQUEST(message, buffer) = (char *) buf;
    MPIDIG_REQUEST(message, count) = count;

    mpi_errno = MPIDI_XPMEM_handle_lmt_coop_recv(unexp_rreq->src_offset,
                                                 unexp_rreq->data_sz,
                                                 MPIDI_XPMEM_REQUEST(message, sreq_ptr),
                                                 unexp_rreq->src_lrank,
                                                 MPIDI_SHM_XPMEM_SEND_LMT_RTS, message);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_HANDLE_HANDLE_UNEXP_RECV);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif /* XPMEM_RECV_H_INCLUDED */
