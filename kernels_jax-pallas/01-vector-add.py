import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def add_vectors_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y


@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)


def main(unused_argv):
    x_val = jnp.arange(512)
    y_val = jnp.arange(512, 1024)
    print(add_vectors(x_val, y_val))


if __name__ == "__main__":
    from absl import app
    app.run(main)
