#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

cudaError_t RunUpdateTextureKernel( uint32_t* rgba, int width, int height );
