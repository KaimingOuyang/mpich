/*
 *  Copyright (C) by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpl.h"
#include "uthash.h"
#include <assert.h>

#define CUDA_ERR_CHECK(ret) if (unlikely((ret) != cudaSuccess)) goto fn_fail
#define CU_ERR_CHECK(ret) if (unlikely((ret) != CUDA_SUCCESS)) goto fn_fail

struct cache_elem {
    int local_dev_id;
    int global_dev_id;
    UT_hash_handle hh;
};

static struct cache_elem *local_to_global_dev_id = NULL;
static struct cache_elem *global_to_local_dev_id = NULL;

static int get_global_dev_id(int local_dev_id)
{
    struct cache_elem *tmp;
    HASH_FIND_INT(local_to_global_dev_id, &local_dev_id, tmp);

    /* local to global device mapping should always succeed */
    assert(tmp);

    return tmp->global_dev_id;
}

static int get_local_dev_id(int global_dev_id)
{
    struct cache_elem *tmp;
    HASH_FIND_INT(global_to_local_dev_id, &global_dev_id, tmp);

    if (tmp)
        return tmp->local_dev_id;
    else
        return -1;
}

typedef struct {
    uintptr_t remote_base_addr;
    uintptr_t mapped_base_addr;
} gpu_ipc_handle_obj_s;

static MPL_gavl_tree_t *cuda_ipc_handle_trees;
static int node_local_size;
static int node_local_rank;
static void gpu_ipc_handle_free(void *ipc_handle);

int MPL_gpu_query_pointer_attr(const void *ptr, MPL_pointer_attr_t * attr)
{
    cudaError_t ret;
    struct cudaPointerAttributes ptr_attr;
    ret = cudaPointerGetAttributes(&ptr_attr, ptr);
    if (ret == cudaSuccess) {
        switch (ptr_attr.type) {
            case cudaMemoryTypeUnregistered:
                attr->type = MPL_GPU_POINTER_UNREGISTERED_HOST;
                attr->device = ptr_attr.device;
                break;
            case cudaMemoryTypeHost:
                attr->type = MPL_GPU_POINTER_REGISTERED_HOST;
                attr->device = ptr_attr.device;
                break;
            case cudaMemoryTypeDevice:
                attr->type = MPL_GPU_POINTER_DEV;
                attr->device = ptr_attr.device;
                break;
            case cudaMemoryTypeManaged:
                attr->type = MPL_GPU_POINTER_MANAGED;
                attr->device = ptr_attr.device;
                break;
        }
    } else if (ret == cudaErrorInvalidValue) {
        attr->type = MPL_GPU_POINTER_UNREGISTERED_HOST;
        attr->device = -1;
    } else {
        goto fn_fail;
    }

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_ipc_handle_create(const void *ptr, MPL_gpu_ipc_mem_handle_t * ipc_handle)
{
    cudaError_t ret;
    CUresult curet;
    CUdeviceptr pbase;
    size_t len;
    struct cudaPointerAttributes ptr_attr;

    curet = cuMemGetAddressRange(&pbase, &len, (CUdeviceptr) ptr);
    CU_ERR_CHECK(curet);

    ipc_handle->remote_base_addr = (uintptr_t) pbase;
    ipc_handle->len = len;
    ipc_handle->node_rank = node_local_rank;
    ret = cudaPointerGetAttributes(&ptr_attr, ptr);
    ipc_handle->global_dev_id = get_global_dev_id(ptr_attr.device);

    ret = cudaIpcGetMemHandle(&ipc_handle->handle, (void *) pbase);
    CUDA_ERR_CHECK(ret);

    ipc_handle->offset = (uintptr_t) ptr - (uintptr_t) pbase;

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_ipc_handle_map(MPL_gpu_ipc_mem_handle_t ipc_handle, MPL_gpu_device_handle_t dev_handle,
                           void **ptr)
{
    cudaError_t ret;
    int prev_devid;
    int mpl_err = MPL_SUCCESS;
    int node_rank;
    void *pbase;
    gpu_ipc_handle_obj_s *handle_obj;

    node_rank = ipc_handle.node_rank;
    mpl_err =
        MPL_gavl_tree_search(cuda_ipc_handle_trees[node_rank],
                             (void *) ipc_handle.remote_base_addr, ipc_handle.len,
                             (void **) &handle_obj);
    if (mpl_err != MPL_SUCCESS)
        goto fn_fail;

    if (handle_obj == NULL) {
        cudaError_t ret;
        int prev_devid;
        void *pbase;

        handle_obj =
            (gpu_ipc_handle_obj_s *) MPL_malloc(sizeof(gpu_ipc_handle_obj_s), MPL_MEM_OTHER);
        assert(handle_obj != NULL);
        cudaGetDevice(&prev_devid);
        cudaSetDevice(dev_handle);
        ret = cudaIpcOpenMemHandle(&pbase, ipc_handle.handle, cudaIpcMemLazyEnablePeerAccess);
        CUDA_ERR_CHECK(ret);

        *ptr = (void *) ((char *) pbase + ipc_handle.offset);

        cudaSetDevice(prev_devid);
        handle_obj->remote_base_addr = ipc_handle.remote_base_addr;
        handle_obj->mapped_base_addr = (uintptr_t) pbase;
        mpl_err =
            MPL_gavl_tree_insert(cuda_ipc_handle_trees[node_rank],
                                 (void *) ipc_handle.remote_base_addr, ipc_handle.len,
                                 (void *) handle_obj);
        if (mpl_err != MPL_SUCCESS)
            goto fn_fail;
    } else {
        *ptr =
            (void *) (ipc_handle.remote_base_addr - handle_obj->remote_base_addr +
                      handle_obj->mapped_base_addr);
    }

  fn_exit:
    return mpl_err;
  fn_fail:
    goto fn_exit;
}

int MPL_gpu_malloc_host(void **ptr, size_t size)
{
    cudaError_t ret;
    ret = cudaMallocHost(ptr, size);
    CUDA_ERR_CHECK(ret);

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_free_host(void *ptr)
{
    cudaError_t ret;
    ret = cudaFreeHost(ptr);
    CUDA_ERR_CHECK(ret);

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_register_host(const void *ptr, size_t size)
{
    cudaError_t ret;
    ret = cudaHostRegister((void *) ptr, size, cudaHostRegisterDefault);
    CUDA_ERR_CHECK(ret);

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_unregister_host(const void *ptr)
{
    cudaError_t ret;
    ret = cudaHostUnregister((void *) ptr);
    CUDA_ERR_CHECK(ret);

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_malloc(void **ptr, size_t size, MPL_gpu_device_handle_t h_device)
{
    int mpl_errno = MPL_SUCCESS;
    int prev_devid;
    cudaError_t ret;
    cudaGetDevice(&prev_devid);
    cudaSetDevice(h_device);
    ret = cudaMalloc(ptr, size);
    CUDA_ERR_CHECK(ret);

  fn_exit:
    cudaSetDevice(prev_devid);
    return mpl_errno;
  fn_fail:
    mpl_errno = MPL_ERR_GPU_INTERNAL;
    goto fn_exit;
}

int MPL_gpu_free(void *ptr)
{
    cudaError_t ret;
    ret = cudaFree(ptr);
    CUDA_ERR_CHECK(ret);

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_init(int local_size, int node_rank, int need_thread_safety, int *max_dev_id_ptr)
{
    int count;
    int mpl_err = MPL_SUCCESS, max_dev_id = -1;
    cudaError_t ret = cudaGetDeviceCount(&count);

    char *visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    int *global_dev_map = (int *) MPL_malloc(count * sizeof(int), MPL_MEM_OTHER);

    if (visible_devices) {
        for (int i = 0; i < count; i++) {
            char *tmp = strtok(visible_devices, ",");
            assert(tmp);
            global_dev_map[i] = atoi(tmp);
            if (global_dev_map[i] > max_dev_id)
                max_dev_id = global_dev_map[i];
            visible_devices = NULL;
        }
    } else {
        max_dev_id = count - 1;
        for (int i = 0; i < count; i++) {
            global_dev_map[i] = i;
        }
    }

    for (int i = 0; i < count; i++) {
        struct cache_elem *elem = (struct cache_elem *) MPL_malloc(sizeof(struct cache_elem),
                                                                   MPL_MEM_OTHER);
        elem->local_dev_id = i;
        elem->global_dev_id = global_dev_map[i];
        HASH_ADD_INT(local_to_global_dev_id, local_dev_id, elem, MPL_MEM_OTHER);
    }

    for (int i = 0; i < count; i++) {
        struct cache_elem *elem = (struct cache_elem *) MPL_malloc(sizeof(struct cache_elem),
                                                                   MPL_MEM_OTHER);
        elem->local_dev_id = i;
        elem->global_dev_id = global_dev_map[i];
        HASH_ADD_INT(global_to_local_dev_id, global_dev_id, elem, MPL_MEM_OTHER);
    }

    *max_dev_id_ptr = max_dev_id;
    MPL_free(global_dev_map);

    node_local_size = local_size;
    node_local_rank = node_rank;
    cuda_ipc_handle_trees = MPL_malloc(sizeof(MPL_gavl_tree_t) * local_size, MPL_MEM_OTHER);
    assert(cuda_ipc_handle_trees != NULL);
    for (int i = 0; i < local_size; ++i) {
        mpl_err = MPL_gavl_tree_create(need_thread_safety, &cuda_ipc_handle_trees[i]);
        if (mpl_err != MPL_SUCCESS) {
            MPL_gpu_finalize();
            break;
        }
    }
    return mpl_err;
}

int MPL_gpu_finalize()
{
    struct cache_elem *current, *tmp;
    HASH_ITER(hh, local_to_global_dev_id, current, tmp) {
        HASH_DEL(local_to_global_dev_id, current);
        MPL_free(current);
    }
    HASH_ITER(hh, global_to_local_dev_id, current, tmp) {
        HASH_DEL(global_to_local_dev_id, current);
        MPL_free(current);
    }

    if (cuda_ipc_handle_trees) {
        for (int i = 0; i < node_local_size; ++i)
            if (cuda_ipc_handle_trees[i])
                MPL_gavl_tree_free(cuda_ipc_handle_trees[i], gpu_ipc_handle_free);
    }
    MPL_free(cuda_ipc_handle_trees);
    return MPL_SUCCESS;
}

int MPL_gpu_ipc_handle_get_local_dev(MPL_gpu_ipc_mem_handle_t ipc_handle,
                                     MPL_gpu_device_handle_t * dev_handle)
{
    *dev_handle = get_local_dev_id(ipc_handle.global_dev_id);
    return MPL_SUCCESS;
}

int MPL_gpu_ipc_handle_get_global_dev_id(MPL_gpu_ipc_mem_handle_t ipc_handle, int *dev_id)
{
    *dev_id = ipc_handle.global_dev_id;
    return MPL_SUCCESS;
}

int MPL_gpu_get_global_visiable_dev(int *dev_map, int len)
{
    char *visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    if (visible_devices) {
        int count;
        cudaError_t ret = cudaGetDeviceCount(&count);
        CUDA_ERR_CHECK(ret);

        memset(dev_map, 0, sizeof(int) * len);
        for (int i = 0; i < count; i++) {
            char *tmp = strtok(visible_devices, ",");
            assert(tmp);
            dev_map[atoi(tmp)] = 1;
            visible_devices = NULL;
        }
    } else {
        for (int i = 0; i < len; ++i)
            dev_map[i] = 1;
    }

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

static void gpu_ipc_handle_free(void *handle_obj)
{
    gpu_ipc_handle_obj_s *handle_obj_ptr = (gpu_ipc_handle_obj_s *) handle_obj;
    cudaIpcCloseMemHandle((void *) handle_obj_ptr->mapped_base_addr);
    MPL_free(handle_obj);
    return;
}
