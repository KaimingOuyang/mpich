#!/bin/bash
# turn on ch4
# link with xpmem

./configure LDFLAGS="-L$HOME/ANL/mpich/src/mpid/ch4/shm/xpmem/lib -lxpmem" --prefix=/home/kaiming/ANL/ch4/ --with-device=ch4:ofi --with-libfabric=/home/kaiming/lib/libfabric --disable-fortran --enable-ch4-netmod-inline=no --enable-ch4-shm-inline=no
