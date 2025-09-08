/*
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "libfabric.h"
#include <vector>
#include "device_host_transport/nvshmem_common_efagda.h"

// EFA GDA specific memory handle (following IBGDA pattern)
typedef struct {
    nvshmemt_libfabric_mem_handle_t libfabric_handle;
    void *buf;
    uint32_t lkey;
    uint32_t rkey;
    bool local_only;
} nvshmemt_efagda_mem_handle_t;
static_assert(sizeof(nvshmemt_efagda_mem_handle_t) <= sizeof(nvshmem_mem_handle_t), "efagda_mem_handle_t size overflow");

typedef struct {
    nvshmemt_libfabric_state_t libfabric_state;
    int log_level;
    struct fi_efa_ops_gda *efa_gda_ops;
    struct fid_ep *ep;
    struct fid_cq *cq_ext;
    struct efa_cq *cuda_cq;
    struct efa_qp *cuda_qp;
    struct cuda_ah_info *cuda_ah;
    int *device_lock;

    // Device-side memory key tracking
    std::vector<nvshmemi_efagda_device_key_t> device_lkeys;
    std::vector<nvshmemi_efagda_device_key_t> device_rkeys;
    nvshmemi_efagda_device_key_t *device_lkeys_d;
    nvshmemi_efagda_device_key_t *device_rkeys_d;

    uint32_t *put_signal_seq_counter;
} nvshmemt_efagda_state_t;
