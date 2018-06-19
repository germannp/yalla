// Boilerplate vector space over data type Pt. Needs members x, y, and z.
#pragma once

#include <type_traits>


template<typename Pt>
struct Is_vector : public std::false_type {};

// float3
__device__ __host__ float3 operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __host__ float3 operator*=(float3& a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

template<>
struct Is_vector<float3> : public std::true_type {};

// float4
__device__ __host__ float4 operator+=(float4& a, const float4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__device__ __host__ float4 operator*=(float4& a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}

template<>
struct Is_vector<float4> : public std::true_type {};


// MAKE_PT(Pt, ...) makes data type Pt with x, y, z, and __VA_ARGS__ as members
#define MEMBER(component) float component
#define COMP_WISE_ADD(component) a.component += b.component
#define COMP_WISE_MULTIPLY(component) a.component *= b

#define MAKE_PT(Pt, ...)                                               \
    struct Pt {                                                        \
        MAP(MEMBER, x, y, z, __VA_ARGS__)                              \
                                                                       \
        friend __device__ __host__ Pt operator+=(Pt& a, const Pt& b)   \
        {                                                              \
            MAP(COMP_WISE_ADD, x, y, z, __VA_ARGS__)                   \
            return a;                                                  \
        }                                                              \
        friend __device__ __host__ Pt operator*=(Pt& a, const float b) \
        {                                                              \
            MAP(COMP_WISE_MULTIPLY, x, y, z, __VA_ARGS__)              \
            return a;                                                  \
        }                                                              \
    };                                                                 \
                                                                       \
    template<>                                                         \
    struct Is_vector<Pt> : public std::true_type {}

// ... where MAP(MACRO, ...) maps MACRO onto __VA_ARGS__, inspired by
// http://stackoverflow.com/questions/11761703/
// clang-format off
#define _MAP0(MACRO, x) MACRO(x);
#define _MAP1(MACRO, x, ...) MACRO(x); _MAP0(MACRO, __VA_ARGS__)
#define _MAP2(MACRO, x, ...) MACRO(x); _MAP1(MACRO, __VA_ARGS__)
#define _MAP3(MACRO, x, ...) MACRO(x); _MAP2(MACRO, __VA_ARGS__)
#define _MAP4(MACRO, x, ...) MACRO(x); _MAP3(MACRO, __VA_ARGS__)
#define _MAP5(MACRO, x, ...) MACRO(x); _MAP4(MACRO, __VA_ARGS__)
#define _MAP6(MACRO, x, ...) MACRO(x); _MAP5(MACRO, __VA_ARGS__)
#define _MAP7(MACRO, x, ...) MACRO(x); _MAP6(MACRO, __VA_ARGS__)
#define _MAP8(MACRO, x, ...) MACRO(x); _MAP7(MACRO, __VA_ARGS__)
#define _MAP9(MACRO, x, ...) MACRO(x); _MAP8(MACRO, __VA_ARGS__)
#define _MAP10(MACRO, x, ...) MACRO(x); _MAP9(MACRO, __VA_ARGS__)
#define _MAP11(MACRO, x, ...) MACRO(x); _MAP10(MACRO, __VA_ARGS__)
#define _MAP12(MACRO, x, ...) MACRO(x); _MAP11(MACRO, __VA_ARGS__)
#define _MAP13(MACRO, x, ...) MACRO(x); _MAP12(MACRO, __VA_ARGS__)
#define _MAP14(MACRO, x, ...) MACRO(x); _MAP13(MACRO, __VA_ARGS__)
#define _MAP15(MACRO, x, ...) MACRO(x); _MAP14(MACRO, __VA_ARGS__)
#define _MAP16(MACRO, x, ...) MACRO(x); _MAP15(MACRO, __VA_ARGS__)
#define _MAP17(MACRO, x, ...) MACRO(x); _MAP16(MACRO, __VA_ARGS__)
#define _MAP18(MACRO, x, ...) MACRO(x); _MAP17(MACRO, __VA_ARGS__)
#define _MAP19(MACRO, x, ...) MACRO(x); _MAP18(MACRO, __VA_ARGS__)
#define _MAP20(MACRO, x, ...) MACRO(x); _MAP19(MACRO, __VA_ARGS__)
#define _MAP21(MACRO, x, ...) MACRO(x); _MAP20(MACRO, __VA_ARGS__)
#define _MAP22(MACRO, x, ...) MACRO(x); _MAP21(MACRO, __VA_ARGS__)
#define _MAP23(MACRO, x, ...) MACRO(x); _MAP22(MACRO, __VA_ARGS__)
#define _MAP24(MACRO, x, ...) MACRO(x); _MAP23(MACRO, __VA_ARGS__)
#define _MAP25(MACRO, x, ...) MACRO(x); _MAP24(MACRO, __VA_ARGS__)
#define _MAP26(MACRO, x, ...) MACRO(x); _MAP25(MACRO, __VA_ARGS__)
#define _MAP27(MACRO, x, ...) MACRO(x); _MAP26(MACRO, __VA_ARGS__)
#define _MAP28(MACRO, x, ...) MACRO(x); _MAP27(MACRO, __VA_ARGS__)
#define _MAP29(MACRO, x, ...) MACRO(x); _MAP28(MACRO, __VA_ARGS__)
#define _MAP30(MACRO, x, ...) MACRO(x); _MAP29(MACRO, __VA_ARGS__)
#define _MAP31(MACRO, x, ...) MACRO(x); _MAP30(MACRO, __VA_ARGS__)
#define _MAP32(MACRO, x, ...) MACRO(x); _MAP31(MACRO, __VA_ARGS__)
#define _MAP33(MACRO, x, ...) MACRO(x); _MAP32(MACRO, __VA_ARGS__)
#define _MAP34(MACRO, x, ...) MACRO(x); _MAP33(MACRO, __VA_ARGS__)
#define _MAP35(MACRO, x, ...) MACRO(x); _MAP34(MACRO, __VA_ARGS__)
#define _MAP36(MACRO, x, ...) MACRO(x); _MAP35(MACRO, __VA_ARGS__)
#define _MAP37(MACRO, x, ...) MACRO(x); _MAP36(MACRO, __VA_ARGS__)
#define _MAP38(MACRO, x, ...) MACRO(x); _MAP37(MACRO, __VA_ARGS__)
#define _MAP39(MACRO, x, ...) MACRO(x); _MAP38(MACRO, __VA_ARGS__)
#define _MAP40(MACRO, x, ...) MACRO(x); _MAP39(MACRO, __VA_ARGS__)
#define _MAP41(MACRO, x, ...) MACRO(x); _MAP40(MACRO, __VA_ARGS__)
#define _MAP42(MACRO, x, ...) MACRO(x); _MAP41(MACRO, __VA_ARGS__)
#define _MAP43(MACRO, x, ...) MACRO(x); _MAP42(MACRO, __VA_ARGS__)
#define _MAP44(MACRO, x, ...) MACRO(x); _MAP43(MACRO, __VA_ARGS__)
#define _MAP45(MACRO, x, ...) MACRO(x); _MAP44(MACRO, __VA_ARGS__)
#define _MAP46(MACRO, x, ...) MACRO(x); _MAP45(MACRO, __VA_ARGS__)
#define _MAP47(MACRO, x, ...) MACRO(x); _MAP46(MACRO, __VA_ARGS__)
#define _MAP48(MACRO, x, ...) MACRO(x); _MAP47(MACRO, __VA_ARGS__)
#define _MAP49(MACRO, x, ...) MACRO(x); _MAP48(MACRO, __VA_ARGS__)
// clang-format on

#define _GET_50th(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,  \
    _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, \
    _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, \
    _44, _45, _46, _47, _48, _49, _50, ...)                                    \
    _50
#define MAP(MACRO, ...)                                                    \
    _GET_50th(__VA_ARGS__, _MAP49, _MAP48, _MAP47, _MAP46, _MAP45, _MAP44, \
        _MAP43, _MAP42, _MAP41, _MAP40, _MAP39, _MAP38, _MAP37, _MAP36,    \
        _MAP35, _MAP34, _MAP33, _MAP32, _MAP31, _MAP30, _MAP29, _MAP28,    \
        _MAP27, _MAP26, _MAP25, _MAP24, _MAP23, _MAP22, _MAP21, _MAP20,    \
        _MAP19, _MAP18, _MAP17, _MAP16, _MAP15, _MAP14, _MAP13, _MAP12,    \
        _MAP11, _MAP10, _MAP9, _MAP8, _MAP7, _MAP6, _MAP5, _MAP4, _MAP3,   \
        _MAP2, _MAP1, _MAP0)(MACRO, __VA_ARGS__)

// Polarized cell
MAKE_PT(Po_cell, theta, phi);


// Generalize += and *= to +, -=, -, *, /= and /
template<typename Pt>
__device__ __host__ typename std::enable_if<Is_vector<Pt>::value, Pt>::type
operator+(const Pt& a, const Pt& b)
{
    auto sum = a;
    sum += b;
    return sum;
}

template<typename Pt>
__device__ __host__ typename std::enable_if<Is_vector<Pt>::value, Pt>::type
operator-=(Pt& a, const Pt& b)
{
    a += -1 * b;
    return a;
}

template<typename Pt>
__device__ __host__ typename std::enable_if<Is_vector<Pt>::value, Pt>::type
operator-(const Pt& a, const Pt& b)
{
    auto diff = a;
    diff -= b;
    return diff;
}

template<typename Pt>
__device__ __host__ typename std::enable_if<Is_vector<Pt>::value, Pt>::type
operator-(const Pt& a)
{
    return -1 * a;
}

template<typename Pt>
__device__ __host__ typename std::enable_if<Is_vector<Pt>::value, Pt>::type
operator*(const Pt& a, const float b)
{
    auto prod = a;
    prod *= b;
    return prod;
}

template<typename Pt>
__device__ __host__ typename std::enable_if<Is_vector<Pt>::value, Pt>::type
operator*(const float b, const Pt& a)
{
    auto prod = a;
    prod *= b;
    return prod;
}

template<typename Pt>
__device__ __host__ typename std::enable_if<Is_vector<Pt>::value, Pt>::type
operator/=(Pt& a, const float b)
{
    a *= 1. / b;
    return a;
}

template<typename Pt>
__device__ __host__ typename std::enable_if<Is_vector<Pt>::value, Pt>::type
operator/(const Pt& a, const float b)
{
    auto quot = a;
    quot /= b;
    return quot;
}
