import torch
import triton
import triton.language as tl
import math

from eas.core.kernels.benchmark.bench import benchmark_all


V1_AUTOTUNE_BASE_CONFIGS = [
    (16, 32, 4),
    (16, 64, 4),
    (32, 32, 4),
    (32, 64, 4),
    (32, 128, 4),
    (64, 32, 4),
    (64, 64, 4),
    (64, 128, 4),
    (128, 32, 4),
    (128, 64, 4),
    (32, 64, 8),
    (64, 64, 8),
]

V1_AUTOTUNE_NUM_STAGES = [2, 3, 4]

V1_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for BLOCK_M, BLOCK_N, num_warps in V1_AUTOTUNE_BASE_CONFIGS
    for num_stages in V1_AUTOTUNE_NUM_STAGES
]


# Q, K, V, output are tensors on the GPU
def attention_torch(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    scores = Q @ K.T / (d**0.5) 
    attn = torch.softmax(scores,dim=-1) 
    output.copy_(attn @ V)


@triton.jit
def FlashAttention1V0(Q,K,V,O, sm_scale : tl.constexpr ,M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,BLOCK_D: tl.constexpr,BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr):

    # LOG2E = 1.4426950408889634
    # qk_scale = sm_scale * LOG2E
    # 加载每个program的q矩阵
    pid = tl.program_id(0) # M 维度
    offset_M = tl.arange(0, BLOCK_M) + pid*BLOCK_M
    offset_D = tl.arange(0, BLOCK_D)

    MASK_Q =  (offset_M[:, None] < M) & (offset_D[None, :] < D)
    MASK_O =  (offset_M[:, None] < M) & (offset_D[None, :] < D)

    Q_ptr = Q + offset_M[:,None]*D + offset_D[None,:]
    O_ptr = O + offset_M[:,None]*D + offset_D[None,:]

    q = tl.load(Q_ptr,mask=MASK_Q,other=0.0)



    # 每行的最大值
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    # 每行的迭代和
    l_i = tl.zeros((BLOCK_M,), tl.float32)    
    # 累加暂存器 [BM,D]      
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    
    for start_n in range(0, N, BLOCK_N):

        # 开始载入每个迭代步的K和V
        offset_N = start_n + tl.arange(0, BLOCK_N)
        MASK_K = (offset_N[:, None] < N) & (offset_D[None, :] < D)
        MASK_V = (offset_N[:, None] < N) & (offset_D[None, :] < D)

        K_ptr = K + offset_N[:, None] * D + offset_D[None,:]
        V_ptr = V + offset_N[:, None] * D + offset_D[None,:]


        k = tl.load(K_ptr,mask=MASK_K,other=0.0)
        v = tl.load(V_ptr,mask=MASK_V,other=0.0)

        # Q @ K_t [BM, BN] 过滤掉最后几个无用的N
        score = tl.dot(q, tl.trans(k)) *  sm_scale
        score = tl.where(offset_N[None, :] < N,score,-float("inf"),)

        # 计算单步 m 和 l
        m_block = tl.max(score, axis=1) # [BM]

        m_new = tl.maximum(m_i, m_block)  # [BM]

        alpha = tl.exp(m_i - m_new) 

        p = tl.exp(score - m_new[:, None]) #  [BM, BN]

        l_i = l_i*alpha + tl.sum(p, axis=1)
        m_i = m_new


         
        # 计算acc 
        acc_block = tl.dot(p.to(v.dtype),v) 
        acc =  acc * alpha[:, None] + acc_block


    acc = acc / l_i[:,None]
    
    tl.store(O_ptr,acc,mask=MASK_O)

     


@triton.jit
def FlashAttention1V1(Q,K,V,O, sm_scale : tl.constexpr ,M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,BLOCK_D: tl.constexpr,BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr):

    LOG2E = 1.4426950408889634
    qk_scale = sm_scale * LOG2E
    # 加载每个program的q矩阵
    pid = tl.program_id(0) # M 维度
    offset_M = tl.arange(0, BLOCK_M) + pid*BLOCK_M
    offset_D = tl.arange(0, BLOCK_D)

    MASK_Q =  (offset_M[:, None] < M) & (offset_D[None, :] < D)
    MASK_O =  (offset_M[:, None] < M) & (offset_D[None, :] < D)

    Q_ptr = Q + offset_M[:,None]*D + offset_D[None,:]
    O_ptr = O + offset_M[:,None]*D + offset_D[None,:]

    q = tl.load(Q_ptr,mask=MASK_Q,other=0.0)



    # 每行的最大值
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    # 每行的迭代和
    l_i = tl.zeros((BLOCK_M,), tl.float32)    
    # 累加暂存器 [BM,D]      
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    
    for start_n in range(0, N, BLOCK_N):

        # 开始载入每个迭代步的K和V
        offset_N = start_n + tl.arange(0, BLOCK_N)
        MASK_K = (offset_N[:, None] < N) & (offset_D[None, :] < D)
        MASK_V = (offset_N[:, None] < N) & (offset_D[None, :] < D)

        K_ptr = K + offset_N[:, None] * D + offset_D[None,:]
        V_ptr = V + offset_N[:, None] * D + offset_D[None,:]


        k = tl.load(K_ptr,mask=MASK_K,other=0.0)
        v = tl.load(V_ptr,mask=MASK_V,other=0.0)

        # Q @ K_t [BM, BN] 过滤掉最后几个无用的N
        score = tl.dot(q, tl.trans(k)) *  qk_scale
        score = tl.where(offset_N[None, :] < N,score,-float("inf"),)

        # 计算单步 m 和 l
        m_block = tl.max(score, axis=1) # [BM]

        m_new = tl.maximum(m_i, m_block)  # [BM]

        alpha = tl.exp2(m_i - m_new) 

        p = tl.exp2(score - m_new[:, None]) #  [BM, BN]

        l_i = l_i*alpha + tl.sum(p, axis=1)
        m_i = m_new


         
        # 计算acc 
        acc_block = tl.dot(p.to(v.dtype),v) 
        acc =  acc * alpha[:, None] + acc_block


    acc = acc / l_i[:,None]
    
    tl.store(O_ptr,acc,mask=MASK_O)

        


@triton.autotune(
    configs=V1_AUTOTUNE_CONFIGS,
    key=["M", "N", "D"],
    warmup=50,
    rep=200,
)
@triton.jit
def FlashAttention1V1Autotuned(Q,K,V,O, sm_scale : tl.constexpr ,M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,BLOCK_D: tl.constexpr,BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr):

    LOG2E = 1.4426950408889634
    qk_scale = sm_scale * LOG2E
    # 加载每个program的q矩阵
    pid = tl.program_id(0) # M 维度
    offset_M = tl.arange(0, BLOCK_M) + pid*BLOCK_M
    offset_D = tl.arange(0, BLOCK_D)

    MASK_Q =  (offset_M[:, None] < M) & (offset_D[None, :] < D)
    MASK_O =  (offset_M[:, None] < M) & (offset_D[None, :] < D)

    Q_ptr = Q + offset_M[:,None]*D + offset_D[None,:]
    O_ptr = O + offset_M[:,None]*D + offset_D[None,:]

    q = tl.load(Q_ptr,mask=MASK_Q,other=0.0)



    # 每行的最大值
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    # 每行的迭代和
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    # 累加暂存器 [BM,D]
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N, BLOCK_N):

        # 开始载入每个迭代步的K和V
        offset_N = start_n + tl.arange(0, BLOCK_N)
        MASK_K = (offset_N[:, None] < N) & (offset_D[None, :] < D)
        MASK_V = (offset_N[:, None] < N) & (offset_D[None, :] < D)

        K_ptr = K + offset_N[:, None] * D + offset_D[None,:]
        V_ptr = V + offset_N[:, None] * D + offset_D[None,:]


        k = tl.load(K_ptr,mask=MASK_K,other=0.0)
        v = tl.load(V_ptr,mask=MASK_V,other=0.0)

        # Q @ K_t [BM, BN] 过滤掉最后几个无用的N
        score = tl.dot(q, tl.trans(k)) *  qk_scale
        score = tl.where(offset_N[None, :] < N,score,-float("inf"),)

        # 计算单步 m 和 l
        m_block = tl.max(score, axis=1) # [BM]

        m_new = tl.maximum(m_i, m_block)  # [BM]

        alpha = tl.exp2(m_i - m_new)

        p = tl.exp2(score - m_new[:, None]) #  [BM, BN]

        l_i = l_i*alpha + tl.sum(p, axis=1)
        m_i = m_new



        # 计算acc
        acc_block = tl.dot(p.to(v.dtype),v)
        acc =  acc * alpha[:, None] + acc_block


    acc = acc / l_i[:,None]

    tl.store(O_ptr,acc,mask=MASK_O)




# Q, K, V, output are tensors on the GPU
def solve_v0(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = max(16, triton.next_power_of_2(d))

    sm_scale = 1.0 / math.sqrt(d)
    # 一个 program 负责 BLOCK_M 行 Q / O
    grid = (
        triton.cdiv(M, BLOCK_M),
    )

    FlashAttention1V0[grid](
        Q,
        K,
        V,
        output,
        sm_scale,
        M,
        N,
        d,
        BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


def solve_v1(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
):
    BLOCK_D = max(16, triton.next_power_of_2(d))

    sm_scale = 1.0 / math.sqrt(d)
    # 一个 program 负责 BLOCK_M 行 Q / O
    grid = (
        triton.cdiv(M, BLOCK_M),
    )

    FlashAttention1V1[grid](
        Q,
        K,
        V,
        output,
        sm_scale,
        M,
        N,
        d,
        BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    solve_v1_autotuned(Q, K, V, output, M, N, d)


def solve_v1_autotuned(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    BLOCK_D = max(16, triton.next_power_of_2(d))

    sm_scale = 1.0 / math.sqrt(d)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
    )

    FlashAttention1V1Autotuned[grid](
        Q,
        K,
        V,
        output,
        sm_scale,
        M,
        N,
        d,
        BLOCK_D=BLOCK_D,
    )


def get_v1_autotune_config():
    best_config = getattr(FlashAttention1V1Autotuned, "best_config", None)
    if best_config is None:
        return None

    return {
        "BLOCK_M": best_config.kwargs["BLOCK_M"],
        "BLOCK_N": best_config.kwargs["BLOCK_N"],
        "num_warps": best_config.num_warps,
        "num_stages": best_config.num_stages,
    }


def print_v1_autotune_config():
    config = get_v1_autotune_config()
    if config is None:
        print("Triton AutoTune Best Config: not selected yet")
        return

    print()
    print("Triton AutoTune Best Config")
    print(
        f"BLOCK_M={config['BLOCK_M']}  "
        f"BLOCK_N={config['BLOCK_N']}  "
        f"num_warps={config['num_warps']}  "
        f"num_stages={config['num_stages']}"
    )


def benchmark_attention(
    M: int = 1024,
    N: int = 1024,
    d: int = 64,
    warmup: int = 50,
    repeat: int = 200,
):
    if not torch.cuda.is_available():
        raise RuntimeError("attention benchmark requires a CUDA device")

    Q = torch.randn((M, d), device="cuda", dtype=torch.float16)
    K = torch.randn((N, d), device="cuda", dtype=torch.float16)
    V = torch.randn((N, d), device="cuda", dtype=torch.float16)

    torch_output = torch.empty((M, d), device="cuda", dtype=torch.float16)
    v0_output = torch.empty((M, d), device="cuda", dtype=torch.float16)
    v1_output = torch.empty((M, d), device="cuda", dtype=torch.float16)
    v1_autotuned_output = torch.empty((M, d), device="cuda", dtype=torch.float16)

    results = benchmark_all(
        {
            "torch": lambda: attention_torch(Q, K, V, torch_output, M, N, d),
            "triton_exp": lambda: solve_v0(Q, K, V, v0_output, M, N, d),
            "triton_exp2": lambda: solve_v1(Q, K, V, v1_output, M, N, d),
            "triton_auto": lambda: solve_v1_autotuned(Q, K, V, v1_autotuned_output, M, N, d),
        },
        warmup=warmup,
        repeat=repeat,
    )
    print_v1_autotune_config()

    return results


if __name__ == "__main__":
    benchmark_attention()
