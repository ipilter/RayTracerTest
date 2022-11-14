#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RenderData.cuh"

cudaError_t RunRenderKernel( rt::RenderData& renderData );
