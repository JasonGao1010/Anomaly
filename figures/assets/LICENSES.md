# Figure design references

The network diagram is drawn in Python in `../draw.py`. Layer stacks, tensor
grids and attention blocks follow the conventions illustrated by ML Visuals.
The layout and network connections are specific to the current model.
The input point cloud is the original STU training scan
`train/206/velodyne/000224.bin`, the middle frame of the 449-scan sequence.
The display retains all measured points within 2.5–35 m range and -3–8 m
height, without label filtering, interpolation, or model inference. Colors
represent height. `scene.png` provides a standalone view of the same points.
Density curves remain schematic.

The motivation diagram is an original mathematical illustration drawn by the
same script. Its bounded support values illustrate the paper's classwise
product and unknown score; they are independent of scans and model outputs.

Reference: https://github.com/dair-ai/ml-visuals
Transformer example: https://github.com/dair-ai/ml-visuals/blob/master/2.png
License: https://github.com/dair-ai/ml-visuals/blob/master/LICENSE

## MIT License

Copyright (c) 2020 dair.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
