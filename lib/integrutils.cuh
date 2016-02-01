// Stuff to integrate float3 points
__device__ void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ float3 zero_Pt(float3* X) {
    float3 zero = {0.0, 0.0, 0.0};
    return zero;
}

__global__ void reset_dX(int n_cells, float3* dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        dX[i].x = 0;
        dX[i].y = 0;
        dX[i].z = 0;
    }
}

__global__ void integrate(int n_cells, float delta_t, const float3* __restrict__ X0,
    float3* X, const float3* __restrict__ dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        X[i].x = X0[i].x + dX[i].x*delta_t;
        X[i].y = X0[i].y + dX[i].y*delta_t;
        X[i].z = X0[i].z + dX[i].z*delta_t;
    }
}

__global__ void integrate(int n_cells, float delta_t, const float3* __restrict__ X0,
    float3* X, const float3* __restrict__ dX, const float3* __restrict__ dX1) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        X[i].x = X0[i].x + (dX[i].x + dX1[i].x)*0.5*delta_t;
        X[i].y = X0[i].y + (dX[i].y + dX1[i].y)*0.5*delta_t;
        X[i].z = X0[i].z + (dX[i].z + dX1[i].z)*0.5*delta_t;
    }
}


// Stuff to integrate float4 points
__device__ void operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__device__ float4 zero_Pt(float4* X) {
    float4 zero = {0.0, 0.0, 0.0, 0.0};
    return zero;
}

__global__ void reset_dX(int n_cells, float4* dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        dX[i].x = 0;
        dX[i].y = 0;
        dX[i].z = 0;
        dX[i].w = 0;
    }
}

__global__ void integrate(int n_cells, float delta_t, const float4* __restrict__ X0,
    float4* X, const float4* __restrict__ dX) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        X[i].x = X0[i].x + dX[i].x*delta_t;
        X[i].y = X0[i].y + dX[i].y*delta_t;
        X[i].z = X0[i].z + dX[i].z*delta_t;
        X[i].w = X0[i].w + dX[i].w*delta_t;
    }
}

__global__ void integrate(int n_cells, float delta_t, const float4* __restrict__ X0,
    float4* X, const float4* __restrict__ dX, const float4* __restrict__ dX1) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_cells) {
        X[i].x = X0[i].x + (dX[i].x + dX1[i].x)*0.5*delta_t;
        X[i].y = X0[i].y + (dX[i].y + dX1[i].y)*0.5*delta_t;
        X[i].z = X0[i].z + (dX[i].z + dX1[i].z)*0.5*delta_t;
        X[i].w = X0[i].w + (dX[i].w + dX1[i].w)*0.5*delta_t;
    }
}
