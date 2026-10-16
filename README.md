# Causal depthwise conv1d in HIP with a PyTorch interface

This repository is based on the following fixed upstream baseline:

- Upstream project: causal_conv1d

- Upstream repository: https://github.com/Dao-AILab/causal-conv1d.git

- Upstream branch: main

- Upstream tag: v1.5.4

- Upstream commit: 22a4577d8ace9d5703daea91a7fb56695492152b

- Upstream license: BSD-3-Clause

Modified by Hygon Information Technology Co., Ltd.

---

Features:
- Support fp32, fp16, bf16.
- Kernel size 2, 3, 4.

## How to compile in dtk environment

Before you compile `causal_conv1d`, you need confirm your environment has `aicc` compiler.

Then lone this repo to your container. Use the following commands to build it.

```bash
cd causal-conv1d
python setup.py bdist_wheel # build. Then goto ./dist to install the whl to python 3rd party library path
```

Test the correctness of `causal conv1d`

```bash
cd tests
pytest test_causal_conv1d.py
```

## How to use

```python
from causal_conv1d import causal_conv1d_fn
```

```python
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
```

Equivalent to:
```python
import torch.nn.functional as F

F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
```