import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


def add_vectors_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]


@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    assert x.shape == y.shape
    BLOCK_SIZE = 2
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    grid = ((x.size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i * BLOCK_SIZE,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i * BLOCK_SIZE,))
        ],
        out_specs=pl.BlockSpec((BLOCK_SIZE,), lambda i: (i * BLOCK_SIZE,)),
    )(x, y)


def main(unused_argv):
    x = jnp.arange(8)
    y = jnp.arange(8, 16)
    z = add_vectors(x, y)

    expected = x + y
    np.testing.assert_allclose(z, expected)


if __name__ == "__main__":
    from absl import app
    app.run(main)
