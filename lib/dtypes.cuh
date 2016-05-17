// Biolerplate vector space over data type Pt


// float3
__device__ __host__ float3 operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ __host__ float3 operator*=(float3& a, const float b) {
    a.x *= b; a.y *= b; a.z *= b;
    return a;
}

// float4
__device__ __host__ float4 operator+=(float4& a, const float4& b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}

__device__ __host__ float4 operator*=(float4& a, const float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
    return a;
}

// Polarized cell
struct pocell {
    float x, y, z, phi, theta;

    friend __device__ pocell operator+=(pocell& a, const pocell& b) {
        a.x += b.x; a.y += b.y; a.z += b.z; a.phi += b.phi; a.theta += b.theta;
        return a;
    }
    friend __device__ pocell operator*=(pocell& a, const float b) {
        a.x *= b; a.y *= b; a.z *= b; a.phi *= b; a.theta *= b;
        return a;
    }
};


// Generalize += and *= to +, -=, -, *, /= and /
template<typename Pt>
__device__ __host__ Pt operator+(const Pt& a, const Pt& b) {
    Pt sum = a;
    sum += b;
    return sum;
}

template<typename Pt>
__device__ __host__ Pt operator-=(Pt& a, const Pt& b) {
    a += -1*b;
    return a;
}

template<typename Pt>
__device__ __host__ Pt operator-(const Pt& a, const Pt& b) {
    Pt diff = a;
    diff -= b;
    return diff;
}

template<typename Pt>
__device__ __host__ Pt operator*(const Pt& a, const float b) {
    Pt prod = a;
    prod *= b;
    return prod;
}

template<typename Pt>
__device__ __host__ Pt operator*(const float b, const Pt& a) {
    Pt prod = a;
    prod *= b;
    return prod;
}

template<typename Pt>
__device__ __host__ Pt operator/=(Pt& a, const float b) {
    a *= 1./b;
    return a;
}

template<typename Pt>
__device__ __host__ Pt operator/(const Pt& a, const float b) {
    Pt quot = a;
    quot /= b;
    return quot;
}
