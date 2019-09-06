/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2019 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef XPMEM_NOINLINE_H_INCLUDED
#define XPMEM_NOINLINE_H_INCLUDED

#include "xpmem_impl.h"

int MPIDI_XPMEM_mpi_init_hook(int rank, int size, int *n_vcis_provided, int *tag_bits);
int MPIDI_XPMEM_mpi_finalize_hook(void);

int MPIDI_XPMEM_mpi_win_create_hook(MPIR_Win * win);
int MPIDI_XPMEM_mpi_win_free_hook(MPIR_Win * win);

int MPIDI_XPMEM_ctrl_do_send_recv_lmt_ack_cb(MPIDI_SHM_ctrl_hdr_t * ctrl_hdr, int send_flag);
#endif /* XPMEM_NOINLINE_H_INCLUDED */
