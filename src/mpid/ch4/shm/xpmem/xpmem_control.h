/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2019 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef XPMEM_CONTROL_H_INCLUDED
#define XPMEM_CONTROL_H_INCLUDED

#include "shm_types.h"
#include "ch4_impl.h"
#include "xpmem_pre.h"
#include "xpmem_impl.h"
#include "xpmem_recv.h"
#include "xpmem_send.h"

int MPIDI_XPMEM_ctrl_send_lmt_rts_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);
int MPIDI_XPMEM_ctrl_send_lmt_cts_req_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);
int MPIDI_XPMEM_ctrl_send_lmt_ack_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);
int MPIDI_XPMEM_ctrl_recv_lmt_ack_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr);

#endif /* XPMEM_CONTROL_H_INCLUDED */
