# LitePT

Standalone LitePT-S architecture from [prs-eth/LitePT](https://github.com/prs-eth/LitePT),
commit `436d04801c8151faebe66a1b2d368a9711e7e6aa`, under the included MIT license.
There are no pretrained weights or datasets in this directory.

`model.py` and `serialization/` come from the upstream `litept/` directory.
`pointrope.py` implements the formula in `libs/pointrope/pointrope_torch.py`:
three independent axis rotations, with the same frequencies and half-vector
rotation. Phases are evaluated directly instead of caching position tables.
Inputs and shared autograd views are never mutated. This avoids the installed
native PointROPE kernel, which cannot execute on the current SM120 GPU.

Other local changes: relative PointROPE import; BF16 attention inputs instead
of FP16 to preserve exponent range; propagation of the serialization shuffle
setting; removal of unused plotting imports. AJAE uses all-return voxel means
as its input and disables stochastic depth and order shuffling for paired scans.
`flash_attn`, `spconv`, and `torch_scatter` are supplied by the existing environment.
The sparse convolution and attention architecture otherwise follows LitePT-S.
These are context features, with no geometric regression supervision.
