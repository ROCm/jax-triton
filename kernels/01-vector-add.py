"""
Vector Addition
"""

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    n_elements = x.size
    BLOCK_SIZE = 16
    grid = (triton.cdiv(x.size, BLOCK_SIZE),)
    return jt.triton_call(
        x,
        y,
        kernel=add_kernel,
        out_shape=out_shape,
        grid=grid,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )


def main(unused_argv):
    x_val = jnp.arange(512)
    y_val = jnp.arange(512, 1024)
    print(add(x_val, y_val))
    print(jax.jit(add)(x_val, y_val))


if __name__ == "__main__":
    from absl import app
    app.run(main)
