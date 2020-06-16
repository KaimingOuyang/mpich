/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpidimpl.h"
#include "gpu_post.h"

int MPIDI_GPU_mpi_init_hook(int rank, int size, int *tag_bits)
{
    int mpl_err, mpi_errno = MPI_SUCCESS;

    mpl_err = MPL_gpu_init();
    if (mpl_err != MPL_SUCCESS)
        mpi_errno = MPI_ERR_OTHER;

    return mpi_errno;
}

int MPIDI_GPU_mpi_finalize_hook(void)
{
    int mpl_err, mpi_errno = MPI_SUCCESS;

    mpl_err = MPL_gpu_finalize();
    if (mpl_err != MPL_SUCCESS)
        mpi_errno = MPI_ERR_OTHER;

    return MPI_SUCCESS;
}
