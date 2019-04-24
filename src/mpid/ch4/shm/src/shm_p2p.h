/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2006 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 */

/*
 * In this file, we directly call the POSIX shared memory module all the time. In the future, this
 * code could have some logic to call other modules in certain circumstances (e.g. XPMEM for large
 * messages and POSIX for small messages).
 */

#ifndef SHM_P2P_H_INCLUDED
#define SHM_P2P_H_INCLUDED

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : MPIR_CVAR_CH4_XPMEM_LMT_MSG_SIZE
      category    : CH4
      type        : int
      default     : 4096
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        If a send message size is larger than MPIR_CVAR_CH4_XPMEM_LMT_MSG_SIZE (in bytes),
        then enable XPMEM-based single copy protocol for intranode communication. The
        environment variable is valid only when then XPMEM shmmod is enabled.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

#include <shm.h>
#ifdef MPIDI_CH4_SHM_ENABLE_XPMEM
#include "../xpmem/shm_inline.h"
#endif
#include "../posix/shm_inline.h"

#define MPIDI_SHM_PT2PT_DEFAULT 1
#define MPIDI_SHM_PT2PT_MULTIMODS 2

/* Enable multi-shmmods protocol when more than one shmmod is enabled. */
#ifdef MPIDI_CH4_SHM_ENABLE_XPMEM
#define MPIDI_SHM_PT2PT_PROT MPIDI_SHM_PT2PT_MULTIMODS
#else
#define MPIDI_SHM_PT2PT_PROT MPIDI_SHM_PT2PT_DEFAULT
#endif

#if (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS)

/* Check if the matched receive request is expected in a shmmod and call
 * corresponding handling routine. If the request is handled by a shmmod,
 * recvd_flag is set to true. The caller should call fallback if no shmmod
 * handles it. */
#undef FUNCNAME
#define FUNCNAME MPIDI_SHM_mmods_try_matched_recv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mmods_try_matched_recv(void *buf,
                                                              MPI_Aint count,
                                                              MPI_Datatype datatype,
                                                              MPIR_Request * message,
                                                              bool * recvd_flag)
{
    int mpi_errno = MPI_SUCCESS, recvtype_iscontig;
    MPIDI_XPMEM_am_unexp_rreq_t *unexp_rreq;
    MPIR_Comm *root_comm = NULL;
    MPIDI_av_entry_t *av = NULL;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MMODS_TRY_MATCHED_RECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MMODS_TRY_MATCHED_RECV);

#ifdef MPIDI_CH4_SHM_ENABLE_XPMEM
    /* XPMEM special receive */
    if (MPIDI_SHM_REQUEST(message, status) & MPIDI_SHM_REQ_XPMEM_SEND_LMT) {
        unexp_rreq = &MPIDI_XPMEM_REQUEST(message, unexp_rreq);
#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
        printf("rank %d - unexpected message impossible\n", MPIDI_XPMEM_global.local_rank);
        fflush(stdout);
        MPIDI_XPMEM_REQUEST(message, call_type) = recv_type;
        MPIR_Datatype_iscontig(datatype, &recvtype_iscontig);
        if (recvtype_iscontig) {
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
        }
#endif

        /* Matching XPMEM LMT receive is now posted */
        MPIR_Datatype_add_ref_if_not_builtin(datatype); /* will -1 once completed in handle_lmt_recv */
        MPIDIG_REQUEST(message, datatype) = datatype;
        MPIDIG_REQUEST(message, buffer) = (char *) buf;
        MPIDIG_REQUEST(message, count) = count;

#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
        mpi_errno = MPIDI_XPMEM_handle_lmt_coop_recv(unexp_rreq->src_offset,
                                                     unexp_rreq->data_sz,
                                                     MPIDI_XPMEM_REQUEST(message, sreq_ptr),
                                                     unexp_rreq->src_lrank,
                                                     unexp_rreq->call_type,
                                                     MPIDI_SHM_XPMEM_SEND_LMT_RTS, message);
#else
        mpi_errno = MPIDI_XPMEM_handle_lmt_recv(unexp_rreq->src_offset,
                                                unexp_rreq->data_sz, unexp_rreq->sreq_ptr,
                                                unexp_rreq->src_lrank, message);
#endif
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        *recvd_flag = true;
    }
#endif /* end of MPIDI_CH4_SHM_ENABLE_XPMEM */

#if defined(MPIDI_CH4_SHM_ENABLE_XPMEM) /* labels are used only when specific shmmods are enabled. */
  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MMODS_TRY_MATCHED_RECV);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
#else
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MMODS_TRY_MATCHED_RECV);
    return mpi_errno;
#endif /* end of MPIDI_CH4_SHM_ENABLE_XPMEM */
}
#endif /* end of (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS) */

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_send(const void *buf, MPI_Aint count,
                                                MPI_Datatype datatype, int rank, int tag,
                                                MPIR_Comm * comm, int context_offset,
                                                MPIDI_av_entry_t * addr, MPIR_Request ** request)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_SEND);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_SEND);

    ret = MPIDI_SHM_mpi_isend(buf, count, datatype, rank, tag, comm, context_offset, addr, request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_SEND);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_ssend(const void *buf, MPI_Aint count,
                                                 MPI_Datatype datatype, int rank, int tag,
                                                 MPIR_Comm * comm, int context_offset,
                                                 MPIDI_av_entry_t * addr, MPIR_Request ** request)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_SSEND);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_SSEND);

    ret =
        MPIDI_POSIX_mpi_ssend(buf, count, datatype, rank, tag, comm, context_offset, addr, request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_SSEND);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_send_init(const void *buf, int count,
                                                     MPI_Datatype datatype, int rank, int tag,
                                                     MPIR_Comm * comm, int context_offset,
                                                     MPIDI_av_entry_t * addr,
                                                     MPIR_Request ** request)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_SEND_INIT);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_SEND_INIT);

    ret = MPIDI_POSIX_mpi_send_init(buf, count, datatype, rank, tag, comm, context_offset, addr,
                                    request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_SEND_INIT);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_ssend_init(const void *buf, int count,
                                                      MPI_Datatype datatype, int rank, int tag,
                                                      MPIR_Comm * comm, int context_offset,
                                                      MPIDI_av_entry_t * addr,
                                                      MPIR_Request ** request)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_SSEND_INIT);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_SSEND_INIT);

    ret = MPIDI_POSIX_mpi_ssend_init(buf, count, datatype, rank, tag, comm, context_offset, addr,
                                     request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_SSEND_INIT);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_rsend_init(const void *buf, int count,
                                                      MPI_Datatype datatype, int rank, int tag,
                                                      MPIR_Comm * comm, int context_offset,
                                                      MPIDI_av_entry_t * addr,
                                                      MPIR_Request ** request)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_RSEND_INIT);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_RSEND_INIT);

    ret = MPIDI_POSIX_mpi_rsend_init(buf, count, datatype, rank, tag, comm, context_offset, addr,
                                     request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_RSEND_INIT);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_bsend_init(const void *buf, int count,
                                                      MPI_Datatype datatype, int rank, int tag,
                                                      MPIR_Comm * comm, int context_offset,
                                                      MPIDI_av_entry_t * addr,
                                                      MPIR_Request ** request)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_BSEND_INIT);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_BSEND_INIT);

    ret = MPIDI_POSIX_mpi_bsend_init(buf, count, datatype, rank, tag, comm, context_offset, addr,
                                     request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_BSEND_INIT);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_isend(const void *buf, MPI_Aint count,
                                                 MPI_Datatype datatype, int rank, int tag,
                                                 MPIR_Comm * comm, int context_offset,
                                                 MPIDI_av_entry_t * addr, MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_ISEND);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_ISEND);

#if (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS)
#ifdef MPIDI_CH4_SHM_ENABLE_XPMEM
    bool dt_contig;
    size_t data_sz;

    MPIDI_Datatype_check_contig_size(datatype, count, dt_contig, data_sz);
    if (dt_contig && data_sz > MPIR_CVAR_CH4_XPMEM_LMT_MSG_SIZE) {
#ifdef MPIDI_CH4_SHM_XPMEM_COOP_P2P
        MPIR_Datatype_add_ref_if_not_builtin(datatype);
        mpi_errno = MPIDI_XPMEM_lmt_coop_isend(buf, count, datatype, rank, tag, comm,
                                               context_offset, addr, MPIDI_SHM_XPMEM_SEND_LMT_RTS,
                                               request);
#else
        mpi_errno = MPIDI_XPMEM_lmt_isend(buf, count, datatype, rank, tag, comm,
                                          context_offset, addr, request);
#endif
        goto fn_exit;
    }
#endif /* end of MPIDI_CH4_SHM_ENABLE_XPMEM */
    mpi_errno = MPIDI_POSIX_mpi_isend(buf, count, datatype, rank, tag, comm,
                                      context_offset, addr, request);
#else /* default */
    mpi_errno = MPIDI_POSIX_mpi_isend(buf, count, datatype, rank, tag, comm,
                                      context_offset, addr, request);
#endif /* end of (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS) */

#if defined(MPIDI_CH4_SHM_ENABLE_XPMEM) /* labels are used only when specific shmmods are enabled. */
  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_ISEND);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
#else
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_ISEND);
    return mpi_errno;
#endif /* end of MPIDI_CH4_SHM_ENABLE_XPMEM */
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_issend(const void *buf, MPI_Aint count,
                                                  MPI_Datatype datatype, int rank, int tag,
                                                  MPIR_Comm * comm, int context_offset,
                                                  MPIDI_av_entry_t * addr, MPIR_Request ** request)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_ISSEND);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_ISSEND);

    ret =
        MPIDI_POSIX_mpi_issend(buf, count, datatype, rank, tag, comm, context_offset, addr,
                               request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_ISSEND);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_cancel_send(MPIR_Request * sreq)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_CANCEL_SEND);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_CANCEL_SEND);

    ret = MPIDI_POSIX_mpi_cancel_send(sreq);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_CANCEL_SEND);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_recv_init(void *buf, int count, MPI_Datatype datatype,
                                                     int rank, int tag, MPIR_Comm * comm,
                                                     int context_offset, MPIR_Request ** request)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_RECV_INIT);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_RECV_INIT);

    ret = MPIDI_POSIX_mpi_recv_init(buf, count, datatype, rank, tag, comm, context_offset, request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_RECV_INIT);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_recv(void *buf, MPI_Aint count, MPI_Datatype datatype,
                                                int rank, int tag, MPIR_Comm * comm,
                                                int context_offset, MPI_Status * status,
                                                MPIR_Request ** request)
{
    int ret = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_RECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_RECV);

    ret = MPIDI_SHM_mpi_irecv(buf, count, datatype, rank, tag, comm, context_offset, request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_RECV);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_irecv(void *buf, MPI_Aint count, MPI_Datatype datatype,
                                                 int rank, int tag, MPIR_Comm * comm,
                                                 int context_offset, MPIR_Request ** request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_IRECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_IRECV);

#if (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS)
    MPIR_Comm *root_comm = NULL;
    MPIR_Request *unexp_req = NULL;
    MPIR_Context_id_t context_id = comm->recvcontext_id + context_offset;

    /* When matches with an unexpected receive, it first tries to receive as
     * a SHM optimized message (e.g., XPMEM SEND LMT). If fails, then receives
     * as CH4 am message. Note that we maintain SHM optimized message in the
     * same unexpected|posted queues as that used by CH4 am messages in order
     * to ensure ordering.
     */

    /* Try to match with an unexpected receive request */
    root_comm = MPIDIG_context_id_to_comm(context_id);
    unexp_req = MPIDIG_dequeue_unexp(rank, tag, context_id, &MPIDIG_COMM(root_comm, unexp_list));

    if (unexp_req) {
        *request = unexp_req;
        /* - Mark as DEQUEUED so that progress engine can complete a matched BUSY
         * rreq once all data arrived;
         * - Mark as IN_PRORESS so that the SHM receive cannot be cancelled. */
        MPIDIG_REQUEST(unexp_req, req->status) |= MPIDIG_REQ_UNEXP_DQUED | MPIDIG_REQ_IN_PROGRESS;
        MPIR_Comm_release(root_comm);   /* -1 for removing from unexp_list */

        mpi_errno = MPIDI_SHM_mpi_imrecv(buf, count, datatype, *request);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);
    } else {
        /* No matching request found, post the receive request  */
        MPIR_Request *rreq = NULL;

        rreq = MPIDIG_request_create(MPIR_REQUEST_KIND__RECV, 2);
        MPIR_ERR_CHKANDSTMT(rreq == NULL, mpi_errno, MPIX_ERR_NOREQ, goto fn_fail, "**nomemreq");

        /* store call type of receive */
        MPIDI_XPMEM_REQUEST(rreq, call_type) = recv_type;

        MPIR_Datatype_add_ref_if_not_builtin(datatype);
        mpi_errno = MPIDIG_prepare_recv_req(rank, tag, context_id, buf, count, datatype, rreq);
        if (mpi_errno)
            MPIR_ERR_POP(mpi_errno);

        MPIR_Comm_add_ref(root_comm);   /* +1 for queuing into posted_list */
        MPIDIG_enqueue_posted(rreq, &MPIDIG_COMM(root_comm, posted_list));

        *request = rreq;
        MPIDI_POSIX_recv_posted_hook(*request, rank, comm);
    }
#else /* default */
    mpi_errno = MPIDI_POSIX_mpi_irecv(buf, count, datatype, rank, tag,
                                      comm, context_offset, request);
#endif /* end of (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS) */

#if (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS)
  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_IRECV);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
#else
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_IRECV);
    return mpi_errno;
#endif /* end of MPIDI_SHM_PT2PT_MULTIMODS */
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_imrecv(void *buf, MPI_Aint count, MPI_Datatype datatype,
                                                  MPIR_Request * message)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_IMRECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_IMRECV);

#if (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS)
    bool recvd_flag = false;

    /* Try shmmod specific matched receive */
    mpi_errno = MPIDI_SHM_mmods_try_matched_recv(buf, count, datatype, message, &recvd_flag);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* If not received, then fallback to POSIX matched receive */
    if (!recvd_flag)
        mpi_errno = MPIDI_POSIX_mpi_imrecv(buf, count, datatype, message);
#else /* default */
    mpi_errno = MPIDI_POSIX_mpi_imrecv(buf, count, datatype, message);
#endif /* end of (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS) */

#if (MPIDI_SHM_PT2PT_PROT == MPIDI_SHM_PT2PT_MULTIMODS)
  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_IMRECV);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
#else
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_IMRECV);
    return mpi_errno;
#endif /* end of MPIDI_SHM_PT2PT_MULTIMODS */
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_cancel_recv(MPIR_Request * rreq)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_CANCEL_RECV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_CANCEL_RECV);

    ret = MPIDI_POSIX_mpi_cancel_recv(rreq);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_CANCEL_RECV);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_improbe(int source, int tag, MPIR_Comm * comm,
                                                   int context_offset, int *flag,
                                                   MPIR_Request ** message, MPI_Status * status)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_IMPROBE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_IMPROBE);

    ret = MPIDI_POSIX_mpi_improbe(source, tag, comm, context_offset, flag, message, status);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_IMPROBE);
    return ret;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_SHM_mpi_iprobe(int source, int tag, MPIR_Comm * comm,
                                                  int context_offset, int *flag,
                                                  MPI_Status * status)
{
    int ret;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_SHM_MPI_IPROBE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_SHM_MPI_IPROBE);

    ret = MPIDI_POSIX_mpi_iprobe(source, tag, comm, context_offset, flag, status);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_SHM_MPI_IPROBE);
    return ret;
}

#endif /* SHM_P2P_H_INCLUDED */
