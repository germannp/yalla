// Stuff to integrate float3 points
__device__ float3 operator+(const float3& a, const float3& b) {
    float3 sum = {a.x + b.x, a.y + b.y, a.z + b.z};
    return sum;
}

__device__ float3 operator*(const float3& a, const float b) {
    float3 prod = {a.x*b, a.y*b, a.z*b};
    return prod;
}


// Stuff to integrate float4 points
__device__ float4 operator+(const float4& a, const float4& b) {
    float4 sum = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
    return sum;
}

__device__ float4 operator*(const float4& a, const float b) {
    float4 prod = {a.x*b, a.y*b, a.z*b, a.w*b};
    return prod;
}
