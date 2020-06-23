/*
 *  Copyright (C) by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpl.h"
#include <assert.h>
#include <libcuhook.h>

#define CUDA_ERR_CHECK(ret) if (unlikely((ret) != cudaSuccess)) goto fn_fail
#define CU_ERR_CHECK(ret) if (unlikely((ret) != CUDA_SUCCESS)) goto fn_fail
#define MPL_ERR_CHECK(ret) if (unlikely((ret) != MPL_SUCCESS)) goto fn_fail

typedef struct gpu_free_hook {
    void (*free_hook) (void *dptr);
    struct gpu_free_hook *next;
} gpu_free_hook_s;

typedef struct {
    uintptr_t remote_base_addr;
    uintptr_t mapped_base_addr;
} gpu_ipc_handle_obj_s;

enum {
    IPC_HANDLE_VALID = 0,
    IPC_HANDLE_MAP_REQUIRE,
};

static MPL_gavl_tree_t **cuda_ipc_handle_trees;
static MPL_gavl_tree_t *cuda_ipc_handle_valid_trees;
static int node_local_size;
static int node_local_rank;
static int node_device_count;
static gpu_free_hook_s *free_hook_chain = NULL;

static void gpu_ipc_handle_free(void *ipc_handle);
static int gpu_mem_hook_init();
static void gpu_cudafree_hook(void *dptr);
static void gpu_ipc_handle_status_free(void *handle_obj);

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

int MPL_gpu_ipc_handle_create(const void *ptr, int rank, MPL_gpu_ipc_mem_handle_t * ipc_handle)
{
    cudaError_t ret;
    CUresult curet;
    CUdeviceptr pbase;
    int mpl_err = MPL_SUCCESS;
    size_t len;
    MPL_gpu_ipc_mem_handle_t *ipc_handle_ptr;

    curet = cuMemGetAddressRange(&pbase, &len, (CUdeviceptr) ptr);
    CU_ERR_CHECK(curet);

    /* check whether current dev buffer has been cached */
    mpl_err =
        MPL_gavl_tree_search(cuda_ipc_handle_valid_trees[rank], (void *) pbase,
                             (uintptr_t) len, (void **) &ipc_handle_ptr);
    MPL_ERR_CHECK(mpl_err);

    if (ipc_handle_ptr == NULL) {
        ipc_handle_ptr =
            (MPL_gpu_ipc_mem_handle_t *) MPL_malloc(sizeof(MPL_gpu_ipc_mem_handle_t),
                                                    MPL_MEM_OTHER);
        ipc_handle_ptr->handle_status = IPC_HANDLE_MAP_REQUIRE;
        ipc_handle_ptr->remote_base_addr = (uintptr_t) pbase;
        ipc_handle_ptr->len = len;
        ipc_handle_ptr->node_rank = node_local_rank;
        ret = cudaIpcGetMemHandle(&ipc_handle_ptr->handle, (void *) pbase);
        CUDA_ERR_CHECK(ret);
    }

    ipc_handle_ptr->offset = (uintptr_t) ptr - (uintptr_t) pbase;
    *ipc_handle = *ipc_handle_ptr;

    if (ipc_handle_ptr->handle_status != IPC_HANDLE_VALID) {
        ipc_handle_ptr->handle_status = IPC_HANDLE_VALID;
        mpl_err =
            MPL_gavl_tree_insert(cuda_ipc_handle_valid_trees[rank],
                                 (void *) pbase, (uintptr_t) len, ipc_handle_ptr);
        MPL_ERR_CHECK(mpl_err);
    }

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
    gpu_ipc_handle_obj_s *handle_obj = NULL;

    node_rank = ipc_handle.node_rank;

    if (ipc_handle.handle_status == IPC_HANDLE_VALID) {
        /* ipc handle is valid, just reuse buffer */
        mpl_err =
            MPL_gavl_tree_search(cuda_ipc_handle_trees[node_rank][dev_handle],
                                 (void *) ipc_handle.remote_base_addr, ipc_handle.len,
                                 (void **) &handle_obj);
        MPL_ERR_CHECK(mpl_err);
    } else if (ipc_handle.handle_status == IPC_HANDLE_MAP_REQUIRE) {
        for (int i = 0; i < node_device_count; ++i) {
            mpl_err =
                MPL_gavl_tree_delete(cuda_ipc_handle_trees[node_rank][i],
                                     (void *) ipc_handle.remote_base_addr, ipc_handle.len);
            MPL_ERR_CHECK(mpl_err);
        }
    }

    if (handle_obj == NULL) {
        /* need to cache buffer handle */
        handle_obj =
            (gpu_ipc_handle_obj_s *) MPL_malloc(sizeof(gpu_ipc_handle_obj_s), MPL_MEM_OTHER);
        assert(handle_obj != NULL);

        cudaGetDevice(&prev_devid);
        cudaSetDevice(dev_handle);
        ret = cudaIpcOpenMemHandle(&pbase, ipc_handle.handle, cudaIpcMemLazyEnablePeerAccess);
        CUDA_ERR_CHECK(ret);
        cudaSetDevice(prev_devid);

        handle_obj->remote_base_addr = ipc_handle.remote_base_addr;
        handle_obj->mapped_base_addr = (uintptr_t) pbase;
        mpl_err =
            MPL_gavl_tree_insert(cuda_ipc_handle_trees[node_rank][dev_handle],
                                 (void *) ipc_handle.remote_base_addr, ipc_handle.len,
                                 (void *) handle_obj);
        MPL_ERR_CHECK(mpl_err);

        *ptr = (void *) ((char *) pbase + ipc_handle.offset);
    } else {
        /* find cached buffer */
        *ptr = (void *) (ipc_handle.offset + handle_obj->mapped_base_addr);
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

    cuda_ipc_handle_valid_trees =
        (MPL_gavl_tree_t *) MPL_malloc(sizeof(MPL_gavl_tree_t) * local_size, MPL_MEM_OTHER);
    assert(cuda_ipc_handle_valid_trees != NULL);
    memset(cuda_ipc_handle_valid_trees, 0, sizeof(MPL_gavl_tree_t) * local_size);

    for (int i = 0; i < local_size; ++i) {
        cuda_ipc_handle_trees[i] =
            (MPL_gavl_tree_t *) MPL_malloc(sizeof(MPL_gavl_tree_t) * count, MPL_MEM_OTHER);
        assert(cuda_ipc_handle_trees[i]);
        memset(cuda_ipc_handle_trees[i], 0, sizeof(MPL_gavl_tree_t) * count);

        for (int j = 0; j < count; ++j) {
            mpl_err = MPL_gavl_tree_create(gpu_ipc_handle_free, &cuda_ipc_handle_trees[i][j]);
            if (mpl_err != MPL_SUCCESS) {
                MPL_gpu_finalize();
                goto fn_fail;
            }
        }

        mpl_err = MPL_gavl_tree_create(gpu_ipc_handle_status_free, &cuda_ipc_handle_valid_trees[i]);
        if (mpl_err != MPL_SUCCESS) {
            MPL_gpu_finalize();
            goto fn_fail;
        }
    }

    gpu_mem_hook_init();
    MPL_gpu_free_hook_register(gpu_cudafree_hook);

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_finalize()
{
    gpu_free_hook_s *prev;

    if (cuda_ipc_handle_trees) {
        for (int i = 0; i < node_local_size; ++i) {
            for (int j = 0; j < node_device_count; ++j)
                if (cuda_ipc_handle_trees[i][j])
                    MPL_gavl_tree_free(cuda_ipc_handle_trees[i][j]);
            MPL_free(cuda_ipc_handle_trees[i]);
        }
    }
    MPL_free(cuda_ipc_handle_trees);

    if (cuda_ipc_handle_valid_trees) {
        for (int i = 0; i < node_local_size; ++i) {
            if (cuda_ipc_handle_valid_trees[i])
                MPL_gavl_tree_free(cuda_ipc_handle_valid_trees[i]);
        }
    }
    MPL_free(cuda_ipc_handle_valid_trees);

    while (free_hook_chain) {
        prev = free_hook_chain;
        free_hook_chain = free_hook_chain->next;
        MPL_free(prev);
    }

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

static void gpu_ipc_handle_status_free(void *handle_obj)
{
    MPL_free(handle_obj);
    return;
}

static void gpu_free_hooks_cb(void *dptr)
{
    gpu_free_hook_s *current = free_hook_chain;
    while (current) {
        current->free_hook(dptr);
        current = current->next;
    }
    return;
}

static int gpu_mem_hook_init()
{
    cuHookRegisterCallback(CU_HOOK_MEM_FREE, PRE_CALL_HOOK, (void *) gpu_free_hooks_cb);
    return MPL_SUCCESS;
}

static void gpu_cudafree_hook(void *dptr)
{
    cudaError_t ret;
    CUresult curet;
    CUdeviceptr pbase;
    size_t len;
    int mpl_err;

    curet = cuMemGetAddressRange(&pbase, &len, (CUdeviceptr) dptr);
    assert(curet == CUDA_SUCCESS);

    for (int i = 0; i < node_local_size; ++i) {
        mpl_err =
            MPL_gavl_tree_delete(cuda_ipc_handle_valid_trees[i], (void *) pbase, (uintptr_t) len);
        assert(mpl_err == MPL_SUCCESS);
    }
}

int MPL_gpu_free_hook_register(void (*free_hook) (void *dptr))
{
    gpu_free_hook_s *hook_obj = MPL_malloc(sizeof(gpu_free_hook_s), MPL_MEM_OTHER);
    assert(hook_obj);
    hook_obj->free_hook = free_hook;
    hook_obj->next = NULL;
    if (!free_hook_chain)
        free_hook_chain = hook_obj;
    else {
        hook_obj->next = free_hook_chain;
        free_hook_chain = hook_obj;
    }

    return MPL_SUCCESS;
}
