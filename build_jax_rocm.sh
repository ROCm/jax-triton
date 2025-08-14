#!/bin/bash
 
MYDIR=$(dirname "$0")
pushd $MYDIR
 
rm -rf dist/*
ROCM_PATH=$(realpath /opt/rocm)
 
python3 -m pip uninstall jax -y
python3 -m pip uninstall jaxlib jax-rocm60-pjrt jax-rocm60-plugin -y
 
python3 ./build/build.py build \
    --wheels=jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
    --use_clang=true --clang_path=/usr/lib/llvm-18/bin/clang \
    --rocm_version=60 --rocm_path=$ROCM_PATH \
    --bazel_options=--override_repository=xla=../xla \
    --rocm_amdgpu_targets=gfx942 && \
python3 setup.py develop --user && python3 -m pip install dist/*.whl
 
popd
