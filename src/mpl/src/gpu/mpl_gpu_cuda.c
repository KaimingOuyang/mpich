/*
 *  Copyright (C) by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpl.h"
#include <assert.h>

#define CUDA_ERR_CHECK(ret) if (unlikely((ret) != cudaSuccess)) goto fn_fail
#define CU_ERR_CHECK(ret) if (unlikely((ret) != CUDA_SUCCESS)) goto fn_fail

typedef struct {
    uintptr_t remote_base_addr;
    uintptr_t mapped_base_addr;
} gpu_ipc_handle_obj_s;

static MPL_gavl_tree_t **cuda_ipc_handle_trees;
static int node_local_size;
static int node_local_rank;
static int node_device_count;
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
        MPL_gavl_tree_search(cuda_ipc_handle_trees[node_rank][dev_handle],
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
            MPL_gavl_tree_insert(cuda_ipc_handle_trees[node_rank][dev_handle],
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

int MPL_gpu_init(int local_size, int node_rank, int *device_count, int *max_dev_id_ptr)
{
    int count;
    int mpl_err = MPL_SUCCESS, max_dev_id = -1;
    cudaError_t ret = cudaGetDeviceCount(&count);
    CUDA_ERR_CHECK(ret);

    char *visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    if (visible_devices) {
        uintptr_t len = strlen(visible_devices);
        char *devices = MPL_malloc(len + 1, MPL_MEM_OTHER);
        char *free_ptr = devices;
        memcpy(devices, visible_devices, len + 1);
        for (int i = 0; i < count; i++) {
            int global_dev_id;
            char *tmp = strtok(devices, ",");
            assert(tmp);
            global_dev_id = atoi(tmp);
            if (global_dev_id > max_dev_id)
                max_dev_id = global_dev_id;
            devices = NULL;
        }
        MPL_free(free_ptr);
    } else {
        max_dev_id = count - 1;
    }

    *max_dev_id_ptr = max_dev_id;
    *device_count = count;

    node_local_size = local_size;
    node_local_rank = node_rank;
    node_device_count = count;

    cuda_ipc_handle_trees =
        (MPL_gavl_tree_t **) MPL_malloc(sizeof(MPL_gavl_tree_t *) * local_size, MPL_MEM_OTHER);
    assert(cuda_ipc_handle_trees != NULL);
    memset(cuda_ipc_handle_trees, 0, sizeof(MPL_gavl_tree_t *) * local_size);

    for (int i = 0; i < local_size; ++i) {
        cuda_ipc_handle_trees[i] =
            (MPL_gavl_tree_t *) MPL_malloc(sizeof(MPL_gavl_tree_t) * count, MPL_MEM_OTHER);
        assert(cuda_ipc_handle_trees[i]);
        memset(cuda_ipc_handle_trees[i], 0, sizeof(MPL_gavl_tree_t) * count);

        for (int j = 0; j < count; ++j) {
            mpl_err = MPL_gavl_tree_create(gpu_ipc_handle_free, &cuda_ipc_handle_trees[i][j]);
            if (mpl_err != MPL_SUCCESS) {
                MPL_gpu_finalize();
                break;
            }
        }
    }

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_finalize()
{
    if (cuda_ipc_handle_trees) {
        for (int i = 0; i < node_local_size; ++i) {
            for (int j = 0; j < node_device_count; ++j)
                if (cuda_ipc_handle_trees[i][j])
                    MPL_gavl_tree_free(cuda_ipc_handle_trees[i][j]);
            MPL_free(cuda_ipc_handle_trees[i]);
        }
    }
    MPL_free(cuda_ipc_handle_trees);
    return MPL_SUCCESS;
}

int MPL_gpu_get_dev_id(MPL_gpu_device_handle_t dev_handle, int *dev_id)
{
    *dev_id = dev_handle;
    return MPL_SUCCESS;
}

int MPL_gpu_get_dev_handle(int dev_id, MPL_gpu_device_handle_t * dev_handle)
{
    *dev_handle = dev_id;
    return MPL_SUCCESS;
}

int MPL_gpu_get_global_dev_ids(int *global_ids, int count)
{
    char *visible_devices = getenv("CUDA_VISIBLE_DEVICES");

    if (visible_devices) {
        uintptr_t len = strlen(visible_devices);
        char *devices = MPL_malloc(len + 1, MPL_MEM_OTHER);
        char *free_ptr = devices;
        memcpy(devices, visible_devices, len + 1);
        for (int i = 0; i < count; i++) {
            char *tmp = strtok(devices, ",");
            assert(tmp);
            global_ids[i] = atoi(tmp);
            devices = NULL;
        }
        MPL_free(free_ptr);
    } else {
        for (int i = 0; i < count; i++) {
            global_ids[i] = i;
        }
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
