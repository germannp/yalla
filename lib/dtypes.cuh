// Necessities to integrate data type Pt


// float3
__device__ float3 operator+(const float3& a, const float3& b) {
    float3 sum = {a.x + b.x, a.y + b.y, a.z + b.z};
    return sum;
}

__device__ float3 operator*(const float3& a, const float b) {
    float3 prod = {a.x*b, a.y*b, a.z*b};
    return prod;
}


// float4
__device__ float4 operator+(const float4& a, const float4& b) {
    float4 sum = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
    return sum;
}

__device__ float4 operator*(const float4& a, const float b) {
    float4 prod = {a.x*b, a.y*b, a.z*b, a.w*b};
    return prod;
}


// Polarized cell
struct pocell {
    float x, y, z, phi, theta;

    friend __device__ pocell operator+(const pocell& a, const pocell& b) {
        pocell sum = {a.x + b.x, a.y + b.y, a.z + b.z, a.phi + b.phi, a.theta + b.theta};
        return sum;
    }
    friend __device__ pocell operator*(const pocell& a, const float b) {
        pocell prod = {a.x*b, a.y*b, a.z*b, a.phi*b, a.theta*b};
        return prod;
    }
};
