/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef XPMEM_RECV_H_INCLUDED
#define XPMEM_RECV_H_INCLUDED

#include "ch4_impl.h"
#include "xpmem_control.h"
#include "xpmem_pre.h"
#include "xpmem_seg.h"
#include "xpmem_impl.h"

/* Handle and complete a matched XPMEM LMT receive request. Input parameters
 * include send buffer info (see definition in MPIDI_SHM_ctrl_xpmem_send_lmt_req_t)
 * and receive request. */
#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_handle_lmt_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_handle_lmt_recv(uint64_t src_offset, uint64_t src_data_sz,
                                                         uint64_t sreq_ptr, int src_lrank,
                                                         MPIR_Request * rreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_XPMEM_seg_t *seg_ptr = NULL;
    void *attached_sbuf = NULL;
    size_t data_sz, recv_data_sz;
    MPIDI_SHM_ctrl_hdr_t ack_ctrl_hdr;
    MPIDI_SHM_ctrl_xpmem_send_lmt_ack_t *slmt_ack_hdr = &ack_ctrl_hdr.xpmem_slmt_ack;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_HANDLE_LMT_RECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_HANDLE_LMT_RECV);

    mpi_errno = MPIDI_XPMEM_seg_regist(src_lrank, src_data_sz, (void *) src_offset,
                                       &seg_ptr, &attached_sbuf);
    if (mpi_errno != MPI_SUCCESS)
        MPIR_ERR_POP(mpi_errno);

    MPIDI_Datatype_check_size(MPIDIG_REQUEST(rreq, datatype), MPIDIG_REQUEST(rreq, count), data_sz);
    if (src_data_sz > data_sz)
        rreq->status.MPI_ERROR = MPI_ERR_TRUNCATE;

    /* Copy data to receive buffer */
    recv_data_sz = MPL_MIN(src_data_sz, data_sz);
    mpi_errno = MPIR_Localcopy(attached_sbuf, recv_data_sz,
                               MPI_BYTE, (char *) MPIDIG_REQUEST(rreq, buffer),
                               MPIDIG_REQUEST(rreq, count), MPIDIG_REQUEST(rreq, datatype));

    XPMEM_PT2PT_DBG_PRINT("handle_lmt_recv: handle matched rreq %p"
                          "[source %d, tag %d, context_id 0x%x], "
                          "copy dst %p, src %p, bytes %ld\n", rreq, MPIDIG_REQUEST(rreq, rank),
                          MPIDIG_REQUEST(rreq, tag), MPIDIG_REQUEST(rreq, context_id),
                          (char *) MPIDIG_REQUEST(rreq, buffer), attached_sbuf, recv_data_sz);

    mpi_errno = MPIDI_XPMEM_seg_deregist(seg_ptr);
    if (mpi_errno != MPI_SUCCESS)
        MPIR_ERR_POP(mpi_errno);

    /* Set receive status */
    MPIR_STATUS_SET_COUNT(rreq->status, recv_data_sz);
    rreq->status.MPI_SOURCE = MPIDIG_REQUEST(rreq, rank);
    rreq->status.MPI_TAG = MPIDIG_REQUEST(rreq, tag);

    /* Send ack to sender */
    slmt_ack_hdr->sreq_ptr = sreq_ptr;
    mpi_errno =
        MPIDI_SHM_do_ctrl_send(MPIDIG_REQUEST(rreq, rank),
                               MPIDIG_context_id_to_comm(MPIDIG_REQUEST(rreq, context_id)),
                               MPIDI_SHM_XPMEM_SEND_LMT_ACK, &ack_ctrl_hdr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    MPIR_Datatype_release_if_not_builtin(MPIDIG_REQUEST(rreq, datatype));
    MPID_Request_complete(rreq);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_HANDLE_LMT_RECV);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_choose_copy_method
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_choose_copy_method(int stype, int rtype)
{
    if (stype == MPIDI_SEND_TYPE) {
        if (rtype == MPIDI_RECV_TYPE)
            return MPIDI_COOP_COPY;
        else
            return MPIDI_SEND_COPY;
    } else {
        /* MPIDI_ISEND_TYPE */
        if (rtype == MPIDI_RECV_TYPE)
            return MPIDI_RECV_COPY;
        else
            return MPIDI_COOP_COPY;
    }
}

#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
#undef FUNCNAME
#define FUNCNAME MPIDI_XPMEM_handle_lmt_coop_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_XPMEM_handle_lmt_coop_recv(uint64_t src_offset,
                                                              uint64_t src_data_sz,
                                                              uint64_t req_ptr, int src_lrank,
                                                              int call_type, int packet_type,
                                                              MPIR_Request * req)
{
    int mpi_errno = MPI_SUCCESS, dt_contig ATTRIBUTE((unused)), ack_type;
    MPIDI_XPMEM_seg_t *seg_ptr = NULL;
    void *attached_sbuf = NULL;
    void *dest_rbuf = NULL;
    size_t data_sz, recv_data_sz, copy_sz;
    MPI_Aint dt_true_lb;
    MPIR_Datatype *dt_ptr ATTRIBUTE((unused));
    OPA_int_t *counter_ptr;
    int copy_method;
    uint64_t cur_offset;
    int perf_copy = 1;

    MPIDI_SHM_ctrl_hdr_t ack_ctrl_hdr;
    MPIDI_SHM_ctrl_xpmem_send_lmt_ack_t *slmt_ack_hdr = &ack_ctrl_hdr.xpmem_slmt_ack;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_XPMEM_HANDLE_LMT_COOP_RECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_XPMEM_HANDLE_LMT_COOP_COOP_RECV);

    mpi_errno = MPIDI_XPMEM_seg_regist(src_lrank, src_data_sz, (void *) src_offset,
                                       &seg_ptr, &attached_sbuf);
    if (mpi_errno != MPI_SUCCESS)
        MPIR_ERR_POP(mpi_errno);

    MPIDI_Datatype_get_info(MPIDIG_REQUEST(req, count), MPIDIG_REQUEST(req, datatype), dt_contig,
                            data_sz, dt_ptr, dt_true_lb);

    /* Copy data to receive buffer */
    recv_data_sz = MPL_MIN(src_data_sz, data_sz);

    if (packet_type == MPIDI_SHM_XPMEM_SEND_LMT_RTS) {
        copy_method =
            MPIDI_XPMEM_choose_copy_method(call_type, MPIDI_XPMEM_REQUEST(req, call_type));
        /* Receiver get the RTS packet */
        if (src_data_sz > data_sz)
            req->status.MPI_ERROR = MPI_ERR_TRUNCATE;
        /* Set receive status */
        MPIR_STATUS_SET_COUNT(req->status, recv_data_sz);
        req->status.MPI_SOURCE = MPIDIG_REQUEST(req, rank);
        req->status.MPI_TAG = MPIDIG_REQUEST(req, tag);
        dest_rbuf = MPIDIG_REQUEST(req, buffer);
        ack_type = MPIDI_SHM_XPMEM_SEND_LMT_ACK;

        switch (copy_method) {
            case MPIDI_COOP_COPY:
                cur_offset = recv_data_sz >> 1;
                copy_sz = recv_data_sz - (recv_data_sz >> 1);
                printf("receiver %d - perform coop copy %ld\n", MPIDI_XPMEM_global.local_rank, copy_sz);
                fflush(stdout);
                mpi_errno = MPIR_Localcopy((char*)attached_sbuf + cur_offset, copy_sz,
                                           MPI_BYTE, (char*) dest_rbuf + cur_offset, copy_sz, MPI_BYTE);
                break;
            case MPIDI_RECV_COPY:
                copy_sz = recv_data_sz;
                printf("receiver %d - perform single copy %ld\n", MPIDI_XPMEM_global.local_rank, copy_sz);
                fflush(stdout);
                mpi_errno = MPIR_Localcopy(attached_sbuf, copy_sz,
                                           MPI_BYTE, dest_rbuf, copy_sz, MPI_BYTE);
                break;
            default:
                perf_copy = 0;
                break;
        }
    } else {
        /* Sender get the CTS packet */
        copy_method =
            MPIDI_XPMEM_choose_copy_method(MPIDI_XPMEM_REQUEST(req, call_type), call_type);
        dest_rbuf = attached_sbuf;
        attached_sbuf = (void *) (char *) MPIDIG_REQUEST(req, buffer);
        ack_type = MPIDI_SHM_XPMEM_RECV_LMT_ACK;

        switch (copy_method) {
            case MPIDI_COOP_COPY:
                copy_sz = recv_data_sz >> 1;
                printf("sender %d - perform coop copy %ld\n", MPIDI_XPMEM_global.local_rank, copy_sz);
                fflush(stdout);
                mpi_errno = MPIR_Localcopy(attached_sbuf, copy_sz,
                                           MPI_BYTE, dest_rbuf, copy_sz, MPI_BYTE);
                break;
            case MPIDI_SEND_COPY:
                copy_sz = recv_data_sz;
                printf("sender %d - perform single copy %ld\n", MPIDI_XPMEM_global.local_rank, copy_sz);
                fflush(stdout);
                mpi_errno = MPIR_Localcopy(attached_sbuf, copy_sz,
                                           MPI_BYTE, dest_rbuf, copy_sz, MPI_BYTE);
                break;
            default:
                perf_copy = 0;
                break;
        }
    }

    mpi_errno = MPIDI_XPMEM_seg_deregist(seg_ptr);
    if (mpi_errno != MPI_SUCCESS)
        MPIR_ERR_POP(mpi_errno);

    /* Send ack */
    if (perf_copy) {
        slmt_ack_hdr->sreq_ptr = req_ptr;
        mpi_errno = MPIDI_SHM_do_ctrl_send(MPIDIG_REQUEST(req, rank),
                                           MPIDIG_context_id_to_comm(MPIDIG_REQUEST
                                                                     (req, context_id)),
                                           ack_type, &ack_ctrl_hdr);
        if (mpi_errno != MPI_SUCCESS)
            MPIR_ERR_POP(mpi_errno);

        if (copy_method != MPIDI_COOP_COPY) {
            MPIR_Datatype_release_if_not_builtin(MPIDIG_REQUEST(req, datatype));
            MPID_Request_complete(req);
        }
    }

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_XPMEM_HANDLE_LMT_COOP_RECV);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif /* MPIDI_CH4_SHM_XPMEM_COOP_P2P */

#endif /* XPMEM_RECV_H_INCLUDED */
