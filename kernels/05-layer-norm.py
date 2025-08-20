"""
Layer Normalization
"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


def layer_norm_fwd(x, w, b, eps, BLOCK_SIZE):
    # reshape input data into 2D tensor
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    strides = jt.strides_from_shape(x_2d.shape)
    # specify output data shape
    out_shape = [
        jax.ShapeDtypeStruct(shape=x_2d.shape, dtype=x_2d.dtype),
        jax.ShapeDtypeStruct(shape=(M,), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(M,), dtype=jnp.float32)
    ]
    grid = (M,)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    y, mean, rstd = jt.triton_call(
        x_2d, w, b,
        kernel=_layer_norm_fwd_fused,
        out_shape=out_shape,
        grid=grid,
        stride=strides[0],
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    saved_tensors_for_bwd = x, w, b, mean, rstd
    return y, saved_tensors_for_bwd


def layer_norm_bwd(dy, saved_tensors_for_bwd, BLOCK_SIZE):
    x, w, b, m, v = saved_tensors_for_bwd
    # reshape input data into 2D tensor
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    # heuristics for amount of parallel reduction stream for DW/DB
    N = w.shape[0]
    GROUP_SIZE_M = 64
    if N <= 8192: GROUP_SIZE_M = 96
    if N <= 4096: GROUP_SIZE_M = 128
    if N <= 1024: GROUP_SIZE_M = 256

    locks = jnp.zeros(2 * GROUP_SIZE_M, dtype=jnp.int32)
    _dw = jnp.zeros((GROUP_SIZE_M, N), dtype=x_2d.dtype)
    _db = jnp.zeros((GROUP_SIZE_M, N), dtype=x_2d.dtype)

    dw = jnp.empty(N, dtype=x_2d.dtype)
    db = jnp.empty(N, dtype=x_2d.dtype)
    dx = jnp.empty(dy.shape, dtype=x_2d.dtype)


    # Compute dx
    # specify output data shape

    out_shape_dx = 
    grid_dx = (M,)

    # enqueue kernel using forward pass heuristics
    # also compute partial sums for DW and DB
    dx, dw, db = 

    _layer_norm_bwd_dx_fused[(M, )](  #
        dx, dy, _dw, _db, x, w, m, v, locks,  #
        x_2d.stride(0), N,  #
        BLOCK_SIZE_N=BLOCK_SIZE,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
        num_warps=num_warps)






    # Compute dw, db
    # specify output data shape

    out_shape_dwdb = 

    grid_dwdb = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    # accumulate partial sums in separate kernel
    _layer_norm_bwd_dwdb[grid](
        _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
        BLOCK_SIZE_M=32,  #
        BLOCK_SIZE_N=128, num_ctas=1)
    return dx, None, dw, db, None


def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )

    key = random.PRNGKey(0)
    key_x, key_w, key_b, key_dy = random.split(key, 4)

    x = -2.3 + 0.5 * jax.random.normal(key_x, shape=x_shape, dtype=dtype)
    weight = jax.random.uniform(key_w, shape=w_shape, dtype=dtype)
    bias = jax.random.uniform(key_w, shape=w_shape, dtype=dtype)
    dy = 0.1 * jax.random.normal(key_dy, shape=x_shape, dtype=dtype)

    # Less than 64KB per feature: enqueue fused kernel
    element_size = 4 if x.dtype == jnp.float32 else 2
    MAX_FUSED_SIZE = 65536 // element_size
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # forward pass
    y, saved_tensors_for_bwd = layer_norm_fwd(x, weight, bias, eps, BLOCK_SIZE)
    print("y:", y)

    # backward pass
    dx _, dw, db, _ = layer_norm_bwd(dy, saved_tensors_for_bwd, BLOCK_SIZE)
    print("dx:", dx)
    print("dw:", dx)
    print("db:", dx)


def main(unused_argv):
    test_layer_norm(1151, 8192, tl.float16)


if __name__ == "__main__":
    from absl import app
    app.run(main)
