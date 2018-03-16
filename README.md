ya||a
=====

ya||a is yet another parallel agent-based model for morphogenesis, like the following branching process:

![Branching Model](branching.gif)

The models can be compiled using [CUDAs](https://developer.nvidia.com/cuda-downloads) `$ nvcc -std=c++11 -arch=sm_XX model.cu` on Linux and macOS or using `> nvcc --cl-version 2017 -arch=compute_XX model.cu` on Windows 10 without further dependencies. The examples produce [vtk files](http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf) that can be visualized for instance with [ParaView](http://www.paraview.org/). The model  [`examples/springs.cu`](examples/springs.cu) is a good starting point to learn more.
