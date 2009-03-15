/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2008 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef BSCI_H_INCLUDED
#define BSCI_H_INCLUDED

#include "hydra.h"

struct HYD_BSCI_fns {
    HYD_Status(*launch_procs) (void);
    HYD_Status(*get_usize) (int *);
    HYD_Status(*wait_for_completion) (void);
    HYD_Status(*finalize) (void);
};

extern struct HYD_BSCI_fns HYD_BSCI_fns;

HYD_Status HYD_BSCI_init(char *bootstrap);
HYD_Status HYD_BSCI_launch_procs(void);
HYD_Status HYD_BSCI_finalize(void);
HYD_Status HYD_BSCI_get_usize(int *size);
HYD_Status HYD_BSCI_wait_for_completion(void);

/* Each bootstrap server has to expose an initialization function */
HYD_Status HYD_BSCI_ssh_init(void);
HYD_Status HYD_BSCI_fork_init(void);

#endif /* BSCI_H_INCLUDED */
