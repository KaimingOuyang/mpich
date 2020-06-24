/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpl.h"
#include <assert.h>

MPL_SUPPRESS_OSX_HAS_NO_SYMBOLS_WARNING;

#ifdef MPL_HAVE_ZE

/* TODO: implement gval memory handle cache for inte gpu */
typedef struct {
    uintptr_t remote_base_addr;
    uintptr_t mapped_base_addr;
    uintptr_t offset;
} gpu_ipc_handle_obj_s;

static MPL_gavl_tree_t **ze_ipc_handle_trees;
static int node_local_size;
static int node_gpu_num;

static void gpu_ipc_handle_free(void *ipc_handle);

ze_driver_handle_t global_ze_driver_handle;
ze_device_handle_t *global_ze_devices_handle;
int gpu_ze_init_driver();

#define ZE_ERR_CHECK(ret) \
    do { \
        if (unlikely((ret) != ZE_RESULT_SUCCESS)) \
            goto fn_fail; \
    } while (0)

int MPL_gpu_init(int local_size, int node_rank, int need_thread_safety, int *device_count_ptr,
                 int *max_dev_id_ptr)
{
    ze_result_t ret;
    int ret_error, device_count;
    ret_error = gpu_ze_init_driver();
    if (ret_error != MPL_SUCCESS)
        goto fn_fail;

    node_local_size = local_size;
    node_local_rank = node_rank;
    zeDriverGet(&node_gpu_num, &global_ze_driver_handle);
    ret = zeDeviceGet(global_ze_driver_handle, &device_count, NULL);
    ZE_ERR_CHECK(ret);

    *max_dev_id_ptr = *device_count_ptr = device_count;
    global_ze_devices_handle =
        (ze_device_handle_t *) MPL_malloc(sizeof(ze_device_handle_t) * device_count, MPL_MEM_OTHER);
    ret = zeDeviceGet(global_ze_driver_handle, &device_count, global_ze_devices_handle);
    ZE_ERR_CHECK(ret);

    ze_ipc_handle_trees =
        (MPL_gavl_tree_t **) MPL_malloc(sizeof(MPL_gavl_tree_t *) * local_size, MPL_MEM_OTHER);
    assert(ze_ipc_handle_trees != NULL);
    for (int i = 0; i < local_size; ++i) {
        ze_ipc_handle_trees[i] =
            (MPL_gavl_tree_t *) MPL_malloc(sizeof(MPL_gavl_tree_t) * node_gpu_num, MPL_MEM_OTHER);
        assert(ze_ipc_handle_trees[i] != NULL);
        for (int j = 0; j < node_gpu_num; ++j) {
            ret_error = MPL_gavl_tree_create(gpu_ipc_handle_free, &ze_ipc_handle_trees[i][j]);
            if (ret_error != MPL_SUCCESS) {
                MPL_gpu_finalize();
                goto fn_exit;
            }
        }
    }

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return ret_error;
}

/* Loads a global ze driver */
int gpu_ze_init_driver()
{
    uint32_t driver_count = 0;
    ze_result_t ret;
    int ret_error = MPL_SUCCESS;
    ze_driver_handle_t *all_drivers;

    ret = zeDriverGet(&driver_count, NULL);
    ZE_ERR_CHECK(ret);
    if (driver_count == 0) {
        goto fn_fail;
    }

    all_drivers = MPL_malloc(driver_count * sizeof(ze_driver_handle_t), MPL_MEM_OTHER);
    if (all_drivers == NULL) {
        ret_error = MPL_ERR_GPU_NOMEM;
        goto fn_fail;
    }
    ret = zeDriverGet(&driver_count, all_drivers);
    ZE_ERR_CHECK(ret);

    int i, d;
    ze_device_handle_t *all_devices = NULL;
    /* Find a driver instance with a GPU device */
    for (i = 0; i < driver_count; ++i) {
        uint32_t device_count = 0;
        ret = zeDeviceGet(all_drivers[i], &device_count, NULL);
        ZE_ERR_CHECK(ret);
        all_devices = MPL_malloc(device_count * sizeof(ze_device_handle_t), MPL_MEM_OTHER);
        if (all_devices == NULL) {
            ret_error = MPL_ERR_GPU_NOMEM;
            goto fn_fail;
        }
        ret = zeDeviceGet(all_drivers[i], &device_count, all_devices);
        ZE_ERR_CHECK(ret);
        /* Check if the driver supports a gpu */
        for (d = 0; d < device_count; ++d) {
            ze_device_properties_t device_properties;
            ret = zeDeviceGetProperties(all_devices[d], &device_properties);
            ZE_ERR_CHECK(ret);

            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                global_ze_driver_handle = all_drivers[i];
                break;
            }
        }

        MPL_free(all_devices);
        all_devices = NULL;
        if (NULL != global_ze_driver_handle) {
            break;
        }
    }

  fn_exit:
    MPL_free(all_drivers);
    return ret_error;
  fn_fail:
    MPL_free(all_devices);
    /* If error code is already set, preserve it */
    if (ret_error == MPL_SUCCESS)
        ret_error = MPL_ERR_GPU_INTERNAL;
    goto fn_exit;
}

int MPL_gpu_finalize()
{
    if (ze_ipc_handle_trees) {
        for (int i = 0; i < node_local_size; ++i)
            if (ze_ipc_handle_trees[i]) {
                for (int j = 0; j < node_gpu_num; ++j)
                    if (ze_ipc_handle_trees[i][j])
                        MPL_gavl_tree_free(ze_ipc_handle_trees[i]);
                MPL_free(ze_ipc_handle_trees[i]);
            }
    }

    MPL_free(global_ze_devices_handle);
    MPL_free(ze_ipc_handle_trees);
    return MPL_SUCCESS;
}

int MPL_gpu_ipc_handle_create(const void *ptr, int rank, MPL_gpu_ipc_mem_handle_t * ipc_handle)
{
    int mpl_err;
    ze_result_t ret;
    ipc_handle->offset = 0;
    ret = zeDriverGetMemIpcHandle(global_ze_driver_handle, ptr, &ipc_handle->handle);
    ZE_ERR_CHECK(ret);

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_ipc_handle_map(MPL_gpu_ipc_mem_handle_t ipc_handle, MPL_gpu_device_handle_t dev_handle,
                           void **ptr)
{
    int mpl_err = MPL_SUCCESS, dev_id;
    int node_rank;
    gpu_ipc_handle_obj_s *handle_obj;

    dev_id = ipc_handle.dev_id;
    node_rank = ipc_handle.node_rank;

    mpl_err = MPL_gavl_tree_search
        (ze_ipc_handle_trees[node_rank][dev_id], (void *) ipc_handle.remote_base_addr,
         ipc_handle.len, (void **) &handle_obj);
    if (mpl_err != MPL_SUCCESS)
        goto fn_fail;

    if (handle_obj == NULL) {
        ze_result_t ret;
        /* TODO: retrive dev_id for device handle */
        handle_obj =
            (gpu_ipc_handle_obj_s *) MPL_malloc(sizeof(gpu_ipc_handle_obj_s), MPL_MEM_OTHER);
        assert(handle_obj != NULL);

        ret =
            zeDriverOpenMemIpcHandle(global_ze_driver_handle,
                                     global_ze_devices_handle[ipc_handle.global_dev_id],
                                     ipc_handle.handle, ZE_IPC_MEMORY_FLAG_NONE, ptr);
        if (ret != ZE_RESULT_SUCCESS) {
            mpl_err = MPL_ERR_GPU_INTERNAL;
            goto fn_fail;
        }

        handle_obj->remote_base_addr = ipc_handle.remote_base_addr;
        handle_obj->mapped_base_addr = (uintptr_t) * ptr;
        handle_obj->offset = ipc_handle.offset;
        mpl_err =
            MPL_gavl_tree_insert(ze_ipc_handle_trees[node_rank][dev_id],
                                 (void *) ipc_handle.remote_base_addr, ipc_handle.len, handle_obj);
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

int MPL_gpu_query_pointer_attr(const void *ptr, MPL_pointer_attr_t * attr)
{
    ze_result_t ret;
    ze_memory_allocation_properties_t ptr_attr;
    ze_device_handle_t device;
    ze_device_properties_t p_device_properties;
    ret = zeDriverGetMemAllocProperties(global_ze_driver_handle, ptr, &ptr_attr, &device);
    ZE_ERR_CHECK(ret);
    attr->device = device;
    switch (ptr_attr.type) {
        case ZE_MEMORY_TYPE_UNKNOWN:
            attr->type = MPL_GPU_POINTER_UNREGISTERED_HOST;
            break;
        case ZE_MEMORY_TYPE_HOST:
            attr->type = MPL_GPU_POINTER_REGISTERED_HOST;
            break;
        case ZE_MEMORY_TYPE_DEVICE:
            attr->type = MPL_GPU_POINTER_DEV;
            break;
        case ZE_MEMORY_TYPE_SHARED:
            attr->type = MPL_GPU_POINTER_MANAGED;
            break;
        default:
            goto fn_fail;
    }

  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_malloc(void **ptr, size_t size, MPL_gpu_device_handle_t h_device)
{
    int ret;
    size_t mem_alignment;
    ze_device_mem_alloc_desc_t device_desc;
    device_desc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT;
    device_desc.ordinal = 0;    /* We currently support a single memory type */
    device_desc.version = ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT;
    /* Currently ZE ignores this augument and uses an internal alignment
     * value. However, this behavior can change in the future. */
    mem_alignment = 1;
    ret = zeDriverAllocDeviceMem(global_ze_driver_handle, &device_desc,
                                 size, mem_alignment, h_device, ptr);

    ZE_ERR_CHECK(ret);
  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_malloc_host(void **ptr, size_t size)
{
    int ret;
    size_t mem_alignment;
    ze_host_mem_alloc_desc_t host_desc;
    host_desc.flags = ZE_HOST_MEM_ALLOC_FLAG_DEFAULT;
    host_desc.version = ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT;

    /* Currently ZE ignores this augument and uses an internal alignment
     * value. However, this behavior can change in the future. */
    mem_alignment = 1;
    ret = zeDriverAllocHostMem(global_ze_driver_handle, &host_desc, size, mem_alignment, ptr);
    ZE_ERR_CHECK(ret);
  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_free(void *ptr)
{
    int ret;
    ret = zeDriverFreeMem(global_ze_driver_handle, ptr);
    ZE_ERR_CHECK(ret);
  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_free_host(void *ptr)
{
    int ret;
    ret = zeDriverFreeMem(global_ze_driver_handle, ptr);
    ZE_ERR_CHECK(ret);
  fn_exit:
    return MPL_SUCCESS;
  fn_fail:
    return MPL_ERR_GPU_INTERNAL;
}

int MPL_gpu_register_host(const void *ptr, size_t size)
{
    return MPL_SUCCESS;
}

int MPL_gpu_unregister_host(const void *ptr)
{
    return MPL_SUCCESS;
}

int MPL_gpu_get_dev_id(MPL_gpu_device_handle_t dev_handle, int *dev_id)
{
    ze_device_properties_t devproerty;

    zeDeviceGetProperties(dev_handle, &devproerty);
    *dev_id = devproerty.deviceId;
    return MPL_SUCCESS;
}

int MPL_gpu_get_dev_handle(int dev_id, MPL_gpu_device_handle_t * dev_handle)
{
    *dev_handle = device_handles[dev_id];
    return MPL_SUCCESS;
}

int MPL_gpu_get_global_dev_ids(int *global_ids, int count)
{
    for (int i = 0; i < count; ++i)
        global_ids[i] = i;
    return MPL_SUCCESS;
}

static void gpu_ipc_handle_free(void *handle_obj)
{
    gpu_ipc_handle_obj_s *handle_obj_ptr = (gpu_ipc_handle_obj_s *) handle_obj;
    zeDriverCloseMemIpcHandle(global_ze_driver_handle,
                              handle_obj_ptr->mapped_base_addr - handle_obj_ptr->offset);
    MPL_free(handle_obj);
    return;
}

int MPL_gpu_free_hook_register(void (*free_hook) (void *dptr))
{
    return MPL_SUCCESS;
}

int MPL_gpu_ipc_handle_cache(int rank, MPL_gpu_ipc_mem_handle_t ipc_handle)
{
    return MPL_SUCCESS;
}

#endif /* MPL_HAVE_ZE */
