#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RenderData.h"

cudaError_t RunUpdateTextureKernel( rt::RenderData& renderData );
