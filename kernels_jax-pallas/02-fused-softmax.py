import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


def softmax_kernel(x_ref, y_ref):
    row_idx = pl.program_id(0)
    row = x_ref[row_idx, :]
    row_minus_max = row - jnp.max(row)

    numerator = jnp.exp(row_minus_max)
    denominator = jnp.sum(numerator)
    y_ref[row_idx, :] = numerator / denominator


@jax.jit
def softmax(x: jax.Array) -> jax.Array:
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    grid = (x.shape[0],)
    return pl.pallas_call(
        softmax_kernel,
        out_shape=out_shape,
        grid=grid,
    )(x)


def main(unused_argv):
    x = jnp.ones((1024, 512), dtype="float32")
    y = softmax(x)

    expected = jax.nn.softmax(x)
    np.testing.assert_allclose(y, expected)
    print("Test passed!")


if __name__ == "__main__":
    from absl import app
    app.run(main)
