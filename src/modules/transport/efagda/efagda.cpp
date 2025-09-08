/*
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

// TODO Convert every tab into 4 spaces in this file

#include "device_host_transport/nvshmem_common_efagda.h"
#include "device_host_transport/nvshmem_common_efagda.h"
#include "efagda.h"
#include "internal/host_transport/transport.h"
#include "non_abi/nvshmem_version.h"
#include "rdma/fabric.h"
#include "rdma/fi_ext_efa.h"
#include "rdma/fi_cm.h"
#include "transport_common.h"

#include <cuda.h>
#include <unistd.h>

static int nvshmemt_efagda_progress(nvshmem_transport_t transport, int is_proxy) {
    int status;
    status = nvshmemt_libfabric_progress(transport, is_proxy);

    // TODO Sync GPU stuff too
    return status;
}

static int nvshmemt_efagda_quiet(struct nvshmem_transport *tcurr, int pe, int is_proxy) {
    int status = 0;
    nvshmemt_efagda_state_t *efagda_state = (nvshmemt_efagda_state_t *)tcurr->state;

    // First, handle host proxy operations via libfabric
    status = nvshmemt_libfabric_quiet(tcurr, pe, is_proxy);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Failed to quiet libfabric host proxy operations: %d\n", status);
        return status;
    }

    // For EFA GDA, completion queue polling is handled on the GPU side
    // The quiet operation just needs to ensure synchronization between
    // GPU and host operations. Since EFA GDA operations are direct GPU-to-NIC,
    // we don't need to poll the completion queue from the host side.

    // The actual completion tracking is handled by the GPU-side operations
    // through the device-side completion counters and fence/quiet functions
    // in the device code (efagda_device.cuh)

    // For now, we just ensure host proxy operations are complete
    // GPU operations will be synchronized through the device-side fence/quiet

    // Memory barrier to ensure all operations are visible
    __sync_synchronize();

    return status;
}

static int nvshmemt_efagda_show_info(struct nvshmem_transport *transport, int style) {
    NVSHMEMI_ERROR_PRINT("efagda show info not implemented");
    return 0;
}

// Host-side consistency enforcement - now handled on device side like IBGDA
static int nvshmemt_efagda_enforce_cst(struct nvshmem_transport *tcurr) {
    int status;
    status = nvshmemt_libfabric_enforce_cst(tcurr);
    // TODO Do Stuff (Sync Host/Proxy)
    return status;
}

static int nvshmemt_efagda_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf,
                                             size_t length, nvshmem_transport_t t,
                                             bool local_only) {
    nvshmemt_efagda_state_t *efagda_state = (nvshmemt_efagda_state_t *)t->state;
    nvshmemt_efagda_mem_handle_t *handle = (nvshmemt_efagda_mem_handle_t *)mem_handle;
    struct fid_mr* mr;
    int status = 0;

    status = nvshmemt_libfabric_get_mem_handle((nvshmem_mem_handle_t *)&handle->libfabric_handle, buf, length, t, local_only);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "libfabric_get_mem_handle failed: %d\n", status);

    mr = handle->libfabric_handle.hdls[NVSHMEMT_LIBFABRIC_PROXY_EP_IDX].mr;

    handle->buf = buf;
    handle->lkey = (uint32_t) efagda_state->efa_gda_ops->get_mr_lkey(mr);
    handle->rkey = (uint32_t)fi_mr_key(mr);
    handle->local_only = local_only;

    if (!local_only) {
        nvshmemi_efagda_device_state_t *device_state =
            (nvshmemi_efagda_device_state_t *)t->type_specific_shared_state;

        size_t num_elements = length >> t->log2_cumem_granularity;
        size_t chunk_idx = ((char *)buf - (char *)t->heap_base) >> t->log2_cumem_granularity;
        size_t num_chunks = (((char *)buf - (char *)t->heap_base) + length +
                             (1ULL << t->log2_cumem_granularity) - 1) >> t->log2_cumem_granularity;

        if (efagda_state->device_lkeys.size() < num_chunks) {
            efagda_state->device_lkeys.resize(num_chunks);
        }

        for (size_t i = 0; i < num_elements; i++) {
            nvshmemi_efagda_device_key_t dev_key;
            dev_key.key = handle->lkey;
            dev_key.next_addr = (uint64_t)buf + length;
            efagda_state->device_lkeys[chunk_idx + i] = dev_key;
        }

        if (efagda_state->device_lkeys_d) {
            cudaFree(efagda_state->device_lkeys_d);
            efagda_state->device_lkeys_d = nullptr;
        }

        size_t lkeys_array_size = efagda_state->device_lkeys.size() * sizeof(nvshmemi_efagda_device_key_t);
        status = cudaMalloc(&efagda_state->device_lkeys_d, lkeys_array_size);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "cudaMalloc for lkeys failed.\n");

        status = cudaMemcpy(efagda_state->device_lkeys_d, efagda_state->device_lkeys.data(),
                           lkeys_array_size, cudaMemcpyHostToDevice);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaMemcpy for lkeys failed.\n");

        device_state->lkeys = efagda_state->device_lkeys_d;
    }

out:
    return status;
}

static int nvshmemt_efagda_release_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                                 nvshmem_transport_t t) {
    nvshmemt_efagda_mem_handle_t *handle = (nvshmemt_efagda_mem_handle_t *)mem_handle;

    return nvshmemt_libfabric_release_mem_handle((nvshmem_mem_handle_t *)&handle->libfabric_handle, t);
}

static int nvshmemt_efagda_can_reach_peer(int *access,
                                          struct nvshmem_transport_pe_info *peer_info,
                                          nvshmem_transport_t t) {
    int status = 0;
    // TODO Switch to GPU read when implemented
    *access = NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ | NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;
    return status;
}

static int nvshmemt_efagda_create_cuda_cq(void* device_buffer, uint16_t num_sub_cqs,
                                          uint32_t sub_cq_size, uint32_t cqe_size,
                                          struct efa_cq **cuda_cq) {
    cudaError_t cuda_err;

    // Allocate device CQ structure
    efa_cq* d_cq;
    cuda_err = cudaMalloc(&d_cq, sizeof(efa_cq));
    if (cuda_err != cudaSuccess) {
        printf("Failed to allocate device memory for cq: %s\n",
               cudaGetErrorString(cuda_err));
        return -1;
    }

    // Initialize and copy CQ structure
    efa_cq h_cq = {};
    h_cq.buf = (uint8_t *)device_buffer;
    h_cq.entry_size = cqe_size;
    h_cq.num_entries = sub_cq_size;
    h_cq.queue_mask = sub_cq_size - 1;
    h_cq.consumed_cnt = 0;
    h_cq.phase = 1;

    cuda_err = cudaMemcpy(d_cq, &h_cq, sizeof(efa_cq), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_cq);
        printf("Failed to copy cq to device: %s\n",
               cudaGetErrorString(cuda_err));
        return -1;
    }

    *cuda_cq = d_cq;

    return 0;
}

static inline void *get_page_start(const void *addr, size_t page_size)
{
    return (void *)((uintptr_t) addr & ~(page_size - 1));
}

static inline void *get_page_end(const void *addr, size_t page_size)
{
    return (void *)((uintptr_t)get_page_start((const char *)addr
            + page_size, page_size) - 1);
}

static int nvshmemt_efagda_get_dmabuf_fd(void *buf, size_t len,
                                         int *dmabuf_fd, uint64_t *dmabuf_offset) {
    int status = 0;
    CUdeviceptr aligned_ptr;
    CUresult cuda_ret;
    size_t aligned_size;
    size_t host_page_size;
    void *base_addr;
    size_t total_size;
    unsigned long long flags;

    host_page_size = sysconf(_SC_PAGE_SIZE);

    cuda_ret = cuMemGetAddressRange((CUdeviceptr *)&base_addr, &total_size, (CUdeviceptr)buf);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemGetAddressRange failed: %d\n", cuda_ret);
    }

    aligned_ptr = (uintptr_t)get_page_start(base_addr, host_page_size);
    aligned_size = (uintptr_t)get_page_end((void *)((uintptr_t)base_addr + total_size - 1),
                                          host_page_size) - (uintptr_t)aligned_ptr + 1;

    flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;

    cuda_ret = cuMemGetHandleForAddressRange((void *)dmabuf_fd, aligned_ptr, aligned_size,
                                            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);

    if ((cuda_ret == CUDA_ERROR_INVALID_VALUE || cuda_ret == CUDA_ERROR_NOT_SUPPORTED) && flags != 0) {
        INFO(TRANSPORT_LOG_WARN, "cuMemGetHandleForAddressRange failed with flags: %llu, "
             "invalid argument. Retrying with no flags.\n", flags);
        cuda_ret = cuMemGetHandleForAddressRange((void *)dmabuf_fd, aligned_ptr, aligned_size,
                                                 CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    }

    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuMemGetHandleForAddressRange failed: %d\n", cuda_ret);
    }

    *dmabuf_offset = (uintptr_t)buf - (uintptr_t)aligned_ptr;
    return 0;

out:
    return status;
}

static int nvshmemt_efagda_create_cq(nvshmemt_efagda_state_t *efagda_state, uint32_t cq_size,
                                     struct fid_cq **cq_ext, struct efa_cq **cuda_cq) {
    int status = 0;
    int dmabuf_fd;
    void *cq_buffer = NULL;
    uint64_t dmabuf_offset;
    uint32_t cq_entries;
    uint32_t buf_size;
    struct fi_cq_attr cq_attr = { .wait_obj = FI_WAIT_NONE };
    struct fi_efa_cq_attr cq_ext_attr = { 0 };
    struct fi_efa_cq_init_attr efa_cq_init_attr = { 0 };

    cq_attr.format = FI_CQ_FORMAT_MSG;
    cq_attr.wait_obj = FI_WAIT_NONE;
    cq_attr.size = cq_size;

    cq_entries = cq_attr.size;
    buf_size = cq_entries * 32 + 4096;

    status = cudaMalloc(&cq_buffer, buf_size);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "Failed to cudaMalloc CQ buffer\n");

    status = nvshmemt_efagda_get_dmabuf_fd(cq_buffer, buf_size, &dmabuf_fd, &dmabuf_offset);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to get CQ buffer dma fd: %d\n", status);

    efa_cq_init_attr.flags = FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF;
    efa_cq_init_attr.ext_mem_dmabuf.length = buf_size;
    efa_cq_init_attr.ext_mem_dmabuf.offset = dmabuf_offset;
    efa_cq_init_attr.ext_mem_dmabuf.fd = dmabuf_fd;

    if (!efagda_state->efa_gda_ops || !efagda_state->efa_gda_ops->cq_open_ext) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "EFA GDA operations not supported - cq_open_ext function not available. "
                          "Please ensure you have the correct libfabric version with EFA GDA support.\n");
    }

    status = efagda_state->efa_gda_ops->cq_open_ext(efagda_state->libfabric_state.domain, &cq_attr, &efa_cq_init_attr,
                                                    cq_ext, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to open external cq: %d: %s\n", status, fi_strerror(status * -1));

    status = efagda_state->efa_gda_ops->query_cq(*cq_ext, &cq_ext_attr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to query external cq: %d: %s\n", status, fi_strerror(status * -1));

    status = nvshmemt_efagda_create_cuda_cq(cq_buffer, 1, cq_entries, cq_ext_attr.entry_size, cuda_cq);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to create EFAGDA cuda cq: %d\n", status);

    return 0;

out:
    if (cq_buffer) cudaFree(cq_buffer);
    return status;
}


static int nvshmemt_efagda_create_cuda_qp(uint8_t *sq_buffer, uint32_t sq_num_wqes,
                                          uint32_t *sq_db, uint32_t sq_max_batch,
                                          uint8_t *rq_buffer, uint32_t rq_num_wqes,
                                          uint32_t *rq_db, struct efa_qp **cuda_qp) {
    cudaError_t cuda_err;

    // Allocate device QP structure
    efa_qp* d_qp;
    cuda_err = cudaMalloc(&d_qp, sizeof(efa_qp));
    if (cuda_err != cudaSuccess) {
        printf("Failed to allocate device memory for qp: %s\n",
               cudaGetErrorString(cuda_err));
        return -1;
    }

    // Initialize QP structure on host
    efa_qp h_qp = {};

    // Initialize SQ
    h_qp.sq.buf = sq_buffer;
    h_qp.sq.wq.db = sq_db;
    h_qp.sq.wq.max_wqes = sq_num_wqes;
    h_qp.sq.wq.max_batch = sq_max_batch;
    h_qp.sq.wq.queue_mask = sq_num_wqes - 1;  // Assuming max_wqes is power of 2
    h_qp.sq.wq.wqes_pending = 0;
    h_qp.sq.wq.wqes_posted = 0;
    h_qp.sq.wq.wqes_completed = 0;
    h_qp.sq.wq.pc = 0;
    h_qp.sq.wq.phase = 0;
    // TODO: get from args or delete:
    h_qp.sq.max_inline_data = 32;
    h_qp.sq.max_rdma_sges = 2;

    // Initialize RQ
    h_qp.rq.buf = rq_buffer;
    h_qp.rq.wq.db = rq_db;
    h_qp.rq.wq.max_wqes = rq_num_wqes;
    h_qp.rq.wq.max_batch = rq_num_wqes;
    h_qp.rq.wq.queue_mask = rq_num_wqes - 1;  // Assuming max_wqes is power of 2
    h_qp.rq.wq.wqes_pending = 0;
    h_qp.rq.wq.wqes_posted = 0;
    h_qp.rq.wq.wqes_completed = 0;
    h_qp.rq.wq.pc = 0;
    h_qp.rq.wq.phase = 1;

    // Copy QP structure to device
    cuda_err = cudaMemcpy(d_qp, &h_qp, sizeof(efa_qp), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_qp);
        printf("Failed to copy qp to device: %s\n",
               cudaGetErrorString(cuda_err));
        return -1;
    }

    *cuda_qp = d_qp;

    return 0;
}

static int nvshmemt_efagda_create_gda_qp(nvshmemt_efagda_state_t *efagda_state) {
    int status = 0;
    CUresult cuda_ret;
    void *sq_ptr = NULL;
    void *rq_ptr = NULL;
    uint32_t *sq_db = NULL;
    uint32_t *rq_db = NULL;
    struct fi_efa_wq_attr sq_attr = {0};
    struct fi_efa_wq_attr rq_attr = {0};

    status = efagda_state->efa_gda_ops->query_qp_wqs(efagda_state->ep, &sq_attr, &rq_attr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "query_qp_wqs failed: %d: %s\n", status, fi_strerror(status * -1));

    // Register SQ buffer
    cuda_ret = cuMemHostRegister(sq_attr.buffer, sq_attr.num_entries * sq_attr.entry_size,
                                CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "SQ buffer cuMemHostRegister failed: %d\n", cuda_ret);
    }

    cuda_ret = cuMemHostGetDevicePointer((CUdeviceptr *)&sq_ptr, sq_attr.buffer, 0);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "SQ buffer cuMemHostGetDevicePointer failed: %d\n", cuda_ret);
    }

    // Register SQ doorbell
    cuda_ret = cuMemHostRegister(sq_attr.doorbell, 4,
                                CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "SQ doorbell cuMemHostRegister failed: %d\n", cuda_ret);
    }

    cuda_ret = cuMemHostGetDevicePointer((CUdeviceptr *)&sq_db, sq_attr.doorbell, 0);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "SQ doorbell cuMemHostGetDevicePointer failed: %d\n", cuda_ret);
    }

    // Register RQ buffer
    cuda_ret = cuMemHostRegister(rq_attr.buffer, rq_attr.num_entries * rq_attr.entry_size,
                                CU_MEMHOSTREGISTER_DEVICEMAP);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "RQ buffer cuMemHostRegister failed: %d\n", cuda_ret);
    }

    cuda_ret = cuMemHostGetDevicePointer((CUdeviceptr *)&rq_ptr, rq_attr.buffer, 0);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "RQ buffer cuMemHostGetDevicePointer failed: %d\n", cuda_ret);
    }

    // Register RQ doorbell
    cuda_ret = cuMemHostRegister(rq_attr.doorbell, 4,
                                CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "RQ doorbell cuMemHostRegister failed: %d\n", cuda_ret);
    }

    cuda_ret = cuMemHostGetDevicePointer((CUdeviceptr *)&rq_db, rq_attr.doorbell, 0);
    if (cuda_ret != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "RQ doorbell cuMemHostGetDevicePointer failed: %d\n", cuda_ret);
    }

    status = nvshmemt_efagda_create_cuda_qp((uint8_t *)sq_ptr, sq_attr.num_entries, sq_db,
                                           sq_attr.max_batch, (uint8_t *)rq_ptr,
                                           rq_attr.num_entries, rq_db, &efagda_state->cuda_qp);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmemt_efagda_create_cuda_qp failed: %d\n", status);

    return 0;

out:
    return status;
}

static int nvshmemt_efagda_populate_av(nvshmem_transport_t t, nvshmemt_efagda_state_t *efagda_state) {
    int status, i, j;
    cudaError_t cuda_err;
    size_t ep_namelen = NVSHMEMT_LIBFABRIC_EP_LEN;
    nvshmemt_libfabric_ep_name_t *all_ep_names = NULL;
    nvshmemt_libfabric_ep_name_t *local_ep_names = NULL;
    cuda_ah_info l_ah_info;

    all_ep_names =
        (nvshmemt_libfabric_ep_name_t *)calloc(t->n_pes, sizeof(nvshmemt_libfabric_ep_name_t));
    NVSHMEMI_NULL_ERROR_JMP(all_ep_names, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate array of endpoint names.");

    local_ep_names = (nvshmemt_libfabric_ep_name_t *)calloc(NVSHMEMT_LIBFABRIC_DEFAULT_NUM_EPS,
                                                            sizeof(nvshmemt_libfabric_ep_name_t));
    NVSHMEMI_NULL_ERROR_JMP(local_ep_names, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate array of endpoint names.");

    status = fi_getname(&efagda_state->ep->fid, local_ep_names->name, &ep_namelen);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to get name for endpoint: %d: %s\n", status,
                          fi_strerror(status * -1));

    status = t->boot_handle->allgather(local_ep_names, all_ep_names,
                                       sizeof(nvshmemt_libfabric_ep_name_t),
                                       t->boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to gather endpoint names.\n");

    // Add all efagda EP's into libfabric AV
    for (i = 0; i < t->n_pes; i++) {
        status = fi_av_insert(efagda_state->libfabric_state.addresses[NVSHMEMT_LIBFABRIC_PROXY_EP_IDX], &all_ep_names[i], 1, NULL, 0, NULL);
        if (status < 1) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "Unable to insert ep names in address vector: %d: %s\n", status,
                               fi_strerror(status * -1));
        }

        status = NVSHMEMX_SUCCESS;
    }

    // Create CUDA AV, Only copy libfabric host proxy EP's into CUDA AV
    cuda_err = cudaMalloc(&efagda_state->cuda_ah, sizeof(cuda_ah_info) * t->n_pes);

    // Host proxy EP's are every other entry in AV (i.e.. 1, 3, 5 .... n_pes * 2)
    for (i = NVSHMEMT_LIBFABRIC_PROXY_EP_IDX, j = 0; i < t->n_pes * 2; i+=2, j++) {
        status = efagda_state->efa_gda_ops->query_addr(efagda_state->ep,
                                                    i,
                                                    &l_ah_info.ah,
                                                    &l_ah_info.remote_qpn,
                                                    &l_ah_info.remote_qkey);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to query addr: %d: %s\n", status,
                           fi_strerror(status * -1));
        cudaMemcpy(&efagda_state->cuda_ah[j], &l_ah_info, sizeof(l_ah_info), cudaMemcpyHostToDevice);
    }

out:
    free(all_ep_names);
    free(local_ep_names);
    return status;
}

static int nvshmemt_efagda_populate_device_state(nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_efagda_state_t *efagda_state = (nvshmemt_efagda_state_t *)t->state;
    nvshmemi_efagda_device_state_t *device_state = (nvshmemi_efagda_device_state_t *)t->type_specific_shared_state;
    uint32_t initial_value = 0;

    if (!device_state) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "type_specific_shared_state is NULL\n");
    }

    // Initialize using common macro then populate from host state
    nvshmemi_init_efagda_device_state((*device_state));
    device_state->n_pes = t->n_pes;
    device_state->my_pe = t->my_pe;
    device_state->log2_cumem_granularity = t->log2_cumem_granularity;
    device_state->cuda_qp = efagda_state->cuda_qp;
    device_state->cuda_cq = efagda_state->cuda_cq;
    device_state->cuda_ah = efagda_state->cuda_ah;


    status = cudaMalloc(&efagda_state->put_signal_seq_counter, sizeof(uint32_t));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cudaMalloc for put_signal_seq_counter failed.\n");

    status = cudaMemcpy(efagda_state->put_signal_seq_counter, &initial_value, sizeof(uint32_t), cudaMemcpyHostToDevice);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cudaMemcpy for put_signal_seq_counter initialization failed.\n");
    device_state->put_signal_seq_counter = efagda_state->put_signal_seq_counter;
    
    // Allocate lock in device memory
    status = cudaMalloc(&efagda_state->device_lock, sizeof(int));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cudaMalloc for lock failed.\n");
    status = cudaMemset(efagda_state->device_lock, 0, sizeof(int));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cudaMemset for lock failed.\n");
    device_state->lock = efagda_state->device_lock;
    
    INFO(efagda_state->log_level, "EFA GDA: Populated device state with n_pes=%d, my_pe=%d\n", device_state->n_pes, device_state->my_pe);
out:
    return status;
}

static int nvshmemt_efagda_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                                int num_selected_devs) {
    int status = 0;
    nvshmemt_efagda_state_t *efagda_state = (nvshmemt_efagda_state_t *)t->state;
    struct fi_info *hints = NULL;
    struct fi_av_attr av_attr;

    status = nvshmemt_libfabric_connect_endpoints(t, selected_dev_ids, num_selected_devs);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed connect_endpoints() on internal libfabric host proxy provider: %d: %s\n",
                          status, fi_strerror(status * -1));

    status = fi_open_ops(&efagda_state->libfabric_state.domain->fid, FI_EFA_GDA_OPS, 0,
                         (void **)&efagda_state->efa_gda_ops, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to open efa domain ops: %d: %s\n", status, fi_strerror(status * -1));

    if (!efagda_state->efa_gda_ops || !efagda_state->efa_gda_ops->cq_open_ext) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "EFA GDA operations not supported - cq_open_ext function not available\n");
    }

    status = nvshmemt_efagda_create_cq(efagda_state, 32768,
                                       &efagda_state->cq_ext, &efagda_state->cuda_cq);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to create efagda tx cq\n");

    status = fi_endpoint(efagda_state->libfabric_state.domain, efagda_state->libfabric_state.prov_info, &efagda_state->ep, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to allocate endpoint: %d: %s\n", status, fi_strerror(status * -1));

    status = fi_ep_bind(efagda_state->ep, &efagda_state->cq_ext->fid, FI_TRANSMIT | FI_RECV);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to bind endpoint to completion queue: %d: %s\n", status,
                          fi_strerror(status * -1));

    status = fi_ep_bind(efagda_state->ep, &efagda_state->libfabric_state.addresses[NVSHMEMT_LIBFABRIC_PROXY_EP_IDX]->fid, 0);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to bind endpoint to address vector: %d: %s\n", status,
                          fi_strerror(status * -1));

    status = fi_enable(efagda_state->ep);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to enable endpoint: %d: %s\n", status, fi_strerror(status * -1));

    status = nvshmemt_efagda_create_gda_qp(efagda_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to create GDA QP: %d\n", status);

    status = nvshmemt_efagda_populate_av(t, efagda_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to populate AV with efagda endpoints: %d: %s\n", status,
                          fi_strerror(status * -1));

    status = nvshmemt_efagda_populate_device_state(t);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to populate EFAGDA device state: %d\n", status);

    fi_freeinfo(hints);
    return 0;

out:
    if (hints) fi_freeinfo(hints);
    return status;
}

static int nvshmemt_efagda_add_device_remote_mem_handles(nvshmem_transport_t t, int transport_stride,
                                                        nvshmem_mem_handle_t *mem_handles,
                                                        uint64_t heap_offset, size_t size) {
    int status = 0;
    nvshmemt_efagda_state_t *efagda_state = (nvshmemt_efagda_state_t *)t->state;
    nvshmemi_efagda_device_state_t *device_state = (nvshmemi_efagda_device_state_t *)t->type_specific_shared_state;
    int n_pes = t->n_pes;

    size_t num_elements = size >> t->log2_cumem_granularity;
    size_t chunk_idx = heap_offset >> t->log2_cumem_granularity;
    size_t num_chunks = (heap_offset + size + (1ULL << t->log2_cumem_granularity) - 1) >> t->log2_cumem_granularity;

    if (efagda_state->device_rkeys.size() < num_chunks * n_pes) {
        efagda_state->device_rkeys.resize(num_chunks * n_pes);
    }

    for (size_t i = 0; i < num_elements; i++) {
        for (int pe = 0; pe < n_pes; pe++) {
            nvshmemt_efagda_mem_handle_t *handle =
                (nvshmemt_efagda_mem_handle_t *)&mem_handles[pe * transport_stride + t->index];

            nvshmemi_efagda_device_key_t device_key;
            device_key.key = handle->rkey;
            device_key.next_addr = heap_offset + size;

            efagda_state->device_rkeys[(chunk_idx + i) * n_pes + pe] = device_key;
        }
    }

    if (efagda_state->device_rkeys_d) {
        cudaFree(efagda_state->device_rkeys_d);
        efagda_state->device_rkeys_d = nullptr;
    }

    size_t rkeys_array_size = efagda_state->device_rkeys.size() * sizeof(nvshmemi_efagda_device_key_t);
    status = cudaMalloc(&efagda_state->device_rkeys_d, rkeys_array_size);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "cudaMalloc for rkeys failed.\n");

    status = cudaMemcpy(efagda_state->device_rkeys_d, efagda_state->device_rkeys.data(),
                       rkeys_array_size, cudaMemcpyHostToDevice);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemcpy for rkeys failed.\n");

    device_state->rkeys = efagda_state->device_rkeys_d;
    return 0;

out:
    return status;
}

static int nvshmemt_efagda_finalize(nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_efagda_state_t *efagda_state = (nvshmemt_efagda_state_t *)t->state;

    if (efagda_state->cuda_cq)
        cudaFree(efagda_state->cuda_cq);

    if (efagda_state->cuda_qp)
        cudaFree(efagda_state->cuda_qp);

    if (efagda_state->ep) {
        status = fi_close(&efagda_state->ep->fid);
        if (status)
            NVSHMEMI_WARN_PRINT("Unable to close efa_gda ep: %d: %s\n", status, fi_strerror(status * -1));
    }

    if (efagda_state->cq_ext)
        fi_close(&efagda_state->cq_ext->fid);

    if (efagda_state->cuda_ah)
        cudaFree(efagda_state->cuda_ah);
    if (efagda_state->device_lkeys_d)
        cudaFree(efagda_state->device_lkeys_d);
    if (efagda_state->device_rkeys_d)
        cudaFree(efagda_state->device_rkeys_d);
    efagda_state->device_lkeys.clear();
    efagda_state->device_rkeys.clear();

    // Free device lock
    // if (efagda_state->device_lock) {
    //     cudaFree(efagda_state->device_lock);
    //     efagda_state->device_lock = NULL;
    // }

    status = nvshmemt_libfabric_finalize(t);
    return status;
}

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    int status;
    nvshmemt_efagda_state_t *efagda_state = NULL;
    nvshmem_transport_t transport = NULL;

    status = nvshmemt_libfabric_init(t, table, api_version);
    transport = *t;

    efagda_state = (nvshmemt_efagda_state_t *)calloc(1, sizeof(*efagda_state));
    NVSHMEMI_NULL_ERROR_JMP(efagda_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for efagda transport state.");

    efagda_state->libfabric_state = *((nvshmemt_libfabric_state_t *) transport->state);
    free(transport->state);
    transport->state = efagda_state;

    // Initialize device-side memory key tracking
    efagda_state->device_lkeys_d = nullptr;
    efagda_state->device_rkeys_d = nullptr;

    /* Operations offloaded to host-proxy:
     *   1. All atomic operations
     *   2. RX side of put-with-signal operation.
     *   3. get/g/p rma operations   // TODO Should I just Implement this in the GPU?
     */

    transport->host_ops.can_reach_peer = nvshmemt_efagda_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_efagda_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_efagda_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_efagda_release_mem_handle;
    transport->host_ops.fence = nvshmemt_efagda_quiet;
    transport->host_ops.quiet = nvshmemt_efagda_quiet;
    transport->host_ops.finalize = nvshmemt_efagda_finalize;
    transport->host_ops.show_info = nvshmemt_efagda_show_info;
    transport->host_ops.progress = nvshmemt_efagda_progress;
    transport->host_ops.enforce_cst = nvshmemt_efagda_enforce_cst;
    transport->host_ops.add_device_remote_mem_handles =
        nvshmemt_efagda_add_device_remote_mem_handles;
    transport->host_ops.rma = NULL;
    transport->host_ops.put_signal = NULL;
    efagda_state->log_level = efagda_state->libfabric_state.log_level;

    // TODO Make sure that EFA GDA has the pre-reqs to run successfully on instance type, or fail to use this provider.
out:
    return status;
}
