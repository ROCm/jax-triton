import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def add_vectors_kernel(x_ref, y_ref, o_ref):
    i = pl.program_id(0)
    o_ref[i] = x_ref[i] + y_ref[i]


@jax.jit
def add_vectors(x: jax.Array, y: jax.Array, BLOCK_SIZE = 16) -> jax.Array:
    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(lambda i: (i * BLOCK_SIZE,), (BLOCK_SIZE,)),
            pl.BlockSpec(lambda i: (i * BLOCK_SIZE,), (BLOCK_SIZE,))
        ],
        out_specs=pl.BlockSpec(lambda i: (i,), (BLOCK_SIZE,)),
    )(x, y)


def main(unused_argv):
    x_val = jnp.arange(512)
    y_val = jnp.arange(512, 1024)
    print(add_vectors(x_val, y_val))


if __name__ == "__main__":
    from absl import app
    app.run(main)
