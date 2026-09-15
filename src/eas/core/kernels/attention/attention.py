import torch
import triton
import triton.language as tl
import math

from eas.core.kernels.benchmark.bench import benchmark, benchmark_all


V1_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
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

    return benchmark_all(
        {
            "torch": lambda: attention_torch(Q, K, V, torch_output, M, N, d),
            "triton_exp": lambda: solve_v0(Q, K, V, v0_output, M, N, d),
            "triton_exp2": lambda: solve_v1(Q, K, V, v1_output, M, N, d),
            "triton_auto": lambda: solve_v1_autotuned(Q, K, V, v1_autotuned_output, M, N, d),
        },
        warmup=warmup,
        repeat=repeat,
    )


def autotune_attention_v1(
    M: int = 1024,
    N: int = 1024,
    d: int = 64,
    warmup: int = 50,
    repeat: int = 200,
    configs=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("attention autotune requires a CUDA device")

    if configs is None:
        configs = [
            (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"], config.num_warps)
            for config in V1_AUTOTUNE_CONFIGS
        ]

    Q = torch.randn((M, d), device="cuda", dtype=torch.float16)
    K = torch.randn((N, d), device="cuda", dtype=torch.float16)
    V = torch.randn((N, d), device="cuda", dtype=torch.float16)
    torch_output = torch.empty((M, d), device="cuda", dtype=torch.float16)

    attention_torch(Q, K, V, torch_output, M, N, d)
    torch.cuda.synchronize()

    results = []
    for BLOCK_M, BLOCK_N, num_warps in configs:
        output = torch.empty((M, d), device="cuda", dtype=torch.float16)

        def fn(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps):
            solve_v1(Q, K, V, output, M, N, d, BLOCK_M, BLOCK_N, num_warps)

        fn()
        torch.cuda.synchronize()
        if not torch.allclose(torch_output, output, atol=1e-2, rtol=1e-2):
            diff = (torch_output - output).abs()
            raise AssertionError(
                f"config BM={BLOCK_M} BN={BLOCK_N} warps={num_warps} failed correctness: "
                f"max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}"
            )

        result = benchmark(
            name=f"BM{BLOCK_M}_BN{BLOCK_N}_W{num_warps}",
            fn=fn,
            warmup=warmup,
            repeat=repeat,
        )
        results.append((BLOCK_M, BLOCK_N, num_warps, result))

    results.sort(key=lambda x: x[3].median_ms)

    print()
    print("=" * 100)
    print("FlashAttention1V1 Autotune")
    print("=" * 100)
    for BLOCK_M, BLOCK_N, num_warps, result in results:
        print(
            f"BM={BLOCK_M:<3} BN={BLOCK_N:<3} warps={num_warps:<2} "
            f"median={result.median_ms:.4f} ms  "
            f"mean={result.mean_ms:.4f} ms  "
            f"p20={result.p20_ms:.4f} ms  "
            f"p80={result.p80_ms:.4f} ms"
        )

    return results


if __name__ == "__main__":
    benchmark_attention()
