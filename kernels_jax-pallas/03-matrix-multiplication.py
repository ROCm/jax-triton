import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


def matmul_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] @ y_ref[...]


@jax.jit
def matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    assert x.shape[1] == y.shape[0], "Incompatible dimensions"
    M, K = x.shape
    _, N = y.shape

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    out_shape = jax.ShapeDtypeStruct((M, N), x.dtype)
    grid = (M // BLOCK_SIZE_M,
            N // BLOCK_SIZE_N)
    return pl.pallas_call(
        matmul_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE_M, BLOCK_SIZE_K), lambda i, j: (i, j)),
            pl.BlockSpec((BLOCK_SIZE_K, BLOCK_SIZE_N), lambda i, j: (i, j))
        ],
        out_specs=pl.BlockSpec((BLOCK_SIZE_M, BLOCK_SIZE_N), lambda i, j: (i, j)),
    )(x, y)


def main(unused_argv):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(k1, (512, 512), dtype=jnp.float32)
    y = jax.random.normal(k2, (512, 512), dtype=jnp.float32)
    z = matmul(x, y)

    expected = x @ y
    np.testing.assert_allclose(z, expected)
    print("Test passed!")


if __name__ == "__main__":
    from absl import app
    app.run(main)
