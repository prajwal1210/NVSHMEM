/*
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEM_COMMON_EFAGDA_H
#define NVSHMEM_COMMON_EFAGDA_H

#include <stddef.h>
#include <stdint.h>

#if !defined __CUDACC_RTC__
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#else
#include <cuda/std/cstddef>
#include "cuda/std/cstdint"
#include <cuda/std/climits>
#endif

#include "device_host/nvshmem_types.h"
#include "non_abi/device/pt-to-pt/efa_io_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum efagda_wc_opcode {
	EFAGDA_WC_SEND,
	EFAGDA_WC_RDMA_WRITE,
	EFAGDA_WC_RDMA_READ,
/*
 * Set value of EFA_WC_RECV so consumers can test if a completion is a
 * receive by testing (opcode & IBV_WC_RECV).
 */
	EFAGDA_WC_RECV                  = 1 << 7,
	EFAGDA_WC_RECV_RDMA_WITH_IMM,
};

struct efa_cq {
	uint16_t cq_idx;
	uint8_t *buf;
	uint32_t entry_size;
	uint32_t num_entries;
	uint32_t queue_mask;
	uint32_t consumed_cnt;
	int phase;
	uint32_t* db;
};

struct efa_wq {
	uint32_t max_sge;
	uint32_t max_wqes;
	uint32_t queue_mask;
	uint32_t *db;
	uint32_t max_batch;
	uint32_t wqes_pending;
	uint32_t wqes_posted;
	uint32_t wqes_completed;
	/* Producer counter */
	uint32_t pc;
	int phase;
	uint16_t sub_cq_idx;
};

struct efa_rq {
	struct efa_wq wq;
	uint8_t *buf;
	size_t buf_size;
};

struct efa_sq {
	struct efa_wq wq;
	uint8_t *buf;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
	struct efa_io_tx_wqe curr_wqe;
};

struct efa_qp {
	struct efa_sq sq;
	struct efa_rq rq;
};

// Address handle info for CUDA AV
struct cuda_ah_info {
    uint16_t ah;
    uint16_t remote_qpn;
    uint32_t remote_qkey;
};

// Memory key structure for device access
typedef struct {
    uint32_t key;
    uint64_t next_addr;  // end of this address range + 1
} nvshmemi_efagda_device_key_t;

/* Wire data for put-signal gdr staged atomics
 * 32 bytes
 * | 4 type | 2 op | 2 num_writes | 8 signal | 8 target_addr | 4 sequence_count | 4 resv
 */
typedef struct nvshmemt_efagda_signal_op {
    int type;  /* Must be first */
    uint16_t op;
    uint16_t num_writes;
    uint64_t sig_val;
    void* target_addr;
    uint32_t sequence_count;
    uint32_t src_pe;
} nvshmemt_efagda_signal_op_t;
/*  EFA's inline send size is 32 bytes */
static_assert(sizeof(nvshmemt_efagda_signal_op) == 32);

// EFA GDA device state structure
typedef struct {
    int version;
    int log2_cumem_granularity;
    uint32_t n_pes;
    uint32_t my_pe;
    uint32_t *put_signal_seq_counter;
    struct efa_qp *cuda_qp;
    struct efa_cq *cuda_cq;
    struct cuda_ah_info *cuda_ah;
    nvshmemi_efagda_device_key_t *lkeys;
    nvshmemi_efagda_device_key_t *rkeys;
    int *lock;
} nvshmemi_efagda_device_state_t;
/* TODO static assert for exact size of efagda_device_state_t */
static_assert(sizeof(nvshmemi_efagda_device_state_t) <= sizeof(nvshmemi_device_transport_state_t),
              "efagda_device_state_t must be less than device_transport_state_t.");

#define nvshmemi_init_efagda_device_state(state)                            \
    do {                                                                    \
        state.version = (1 << 16) + sizeof(nvshmemi_efagda_device_state_t); \
        state.log2_cumem_granularity = 0;                                   \
        state.n_pes = 0;                                                    \
        state.my_pe = 0;                                                    \
        state.put_signal_seq_counter = NULL;                            \
        state.cuda_qp = NULL;                                               \
        state.cuda_cq = NULL;                                               \
        state.cuda_ah = NULL;                                               \
        state.lkeys = NULL;                                                 \
        state.rkeys = NULL;                                                 \
        state.lock = NULL;                                                  \
    } while (0);

// External declaration of the host-side EFA GDA device state
#if !defined __CUDACC_RTC__
extern nvshmemi_efagda_device_state_t nvshmemi_efagda_device_state;
#endif

// Initialization function for EFA GDA state
#if !defined __CUDACC_RTC__
#define nvshmemi_init_efagda_device_state_standalone(state) \
    nvshmemi_init_efagda_device_state(state)
#endif

// Device constant memory symbol for EFA GDA (separate from IBGDA)
#ifdef __CUDACC__
extern __constant__ nvshmemi_device_transport_state_t nvshmemi_device_transport_state_d;
#endif

#ifdef __cplusplus
}
#endif

#endif // NVSHMEM_COMMON_EFAGDA_H