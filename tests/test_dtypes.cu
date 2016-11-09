#include "../lib/dtypes.cuh"
#include "minunit.cuh"


const char* test_float3() {
    float3 x {1, 2, 3};
    float3 y {5, 4, 3};

    MU_ASSERT("+= float3 float3, x component", (x += y).x == 1 + 5);
    MU_ASSERT("+= float3 float3, y component", (x += y).y == 2 + 4 + 4);
    MU_ASSERT("+= float3 float3, z component", (x += y).z == 3 + 3 + 3 + 3);

    MU_ASSERT("*= float3 float, x component", (y *= 2).x == 5*2);
    MU_ASSERT("*= float3 float, y component", (y *= 2).y == 4*2*2);
    MU_ASSERT("*= float3 float, z component", (y *= 2).z == 3*2*2*2);

    return NULL;
}


const char* test_float4() {
    float4 x {1, 2, 3, 4};
    float4 y {5, 4, 3, 2};

    MU_ASSERT("+= float4 float4, x component", (x += y).x == 1 + 5);
    MU_ASSERT("+= float4 float4, y component", (x += y).y == 2 + 4 + 4);
    MU_ASSERT("+= float4 float4, z component", (x += y).z == 3 + 3 + 3 + 3);
    MU_ASSERT("+= float4 float4, w component", (x += y).w == 4 + 2 + 2 + 2 + 2);

    MU_ASSERT("*= float4 float, x component", (y *= 2).x == 5*2);
    MU_ASSERT("*= float4 float, y component", (y *= 2).y == 4*2*2);
    MU_ASSERT("*= float4 float, z component", (y *= 2).z == 3*2*2*2);
    MU_ASSERT("*= float4 float, w component", (y *= 2).w == 2*2*2*2*2);

    return NULL;
}


MAKE_PT(My_float3, x, y, z);

const char* test_make_pt() {
    My_float3 x {1, 2, 3};
    My_float3 y {5, 4, 3};

    MU_ASSERT("+= My_float3 My_float3, x component", (x += y).x == 1 + 5);
    MU_ASSERT("+= My_float3 My_float3, y component", (x += y).y == 2 + 4 + 4);
    MU_ASSERT("+= My_float3 My_float3, z component", (x += y).z == 3 + 3 + 3 + 3);

    MU_ASSERT("*= My_float3 float, x component", (y *= 2).x == 5*2);
    MU_ASSERT("*= My_float3 float, y component", (y *= 2).y == 4*2*2);
    MU_ASSERT("*= My_float3 float, z component", (y *= 2).z == 3*2*2*2);

    return NULL;
}


const char* test_generalization() {
    float3 x {1, 2, 3};
    float3 y {4, 3, 2};

    MU_ASSERT("+ float3 float3, x component", (x + y).x == 5);
    MU_ASSERT("+ float3 float3, y component", (x + y).y == 5);
    MU_ASSERT("+ float3 float3, z component", (x + y).z == 5);

    MU_ASSERT("- float3 float3, x component", (x - y).x == -3);
    MU_ASSERT("- float3 float3, y component", (x - y).y == -1);
    MU_ASSERT("- float3 float3, z component", (x - y).z == 1);

    MU_ASSERT("* float3 float, x component", (x*3).x == 3);
    MU_ASSERT("* float3 float, y component", (x*3).y == 6);
    MU_ASSERT("* float3 float, z component", (x*3).z == 9);

    MU_ASSERT("* float float3, x component", (3*x).x == 3);
    MU_ASSERT("* float float3, y component", (3*x).y == 6);
    MU_ASSERT("* float float3, z component", (3*x).z == 9);

    MU_ASSERT("/ float3 float, x component", MU_ISCLOSE((x/3).x, 1./3));
    MU_ASSERT("/ float3 float, y component", MU_ISCLOSE((x/3).y, 2./3));
    MU_ASSERT("/ float3 float, z component", MU_ISCLOSE((x/3).z, 1));

    MU_ASSERT("-= float3 float3, x component", (x -= y).x == 1 - 4);
    MU_ASSERT("-= float3 float3, y component", (x -= y).y == 2 - 3 - 3);
    MU_ASSERT("-= float3 float3, z component", (x -= y).z == 3 - 2 - 2 - 2);

    MU_ASSERT("/= float3 float, x component", MU_ISCLOSE((y /= 3).x, 4./3));
    MU_ASSERT("/= float3 float, y component", MU_ISCLOSE((y /= 3).y, 3./9));
    MU_ASSERT("/= float3 float, z component", MU_ISCLOSE((y /= 3).z, 2./27));

    return NULL;
}


const char* all_tests() {
    MU_RUN_TEST(test_float3);
    MU_RUN_TEST(test_float4);
    MU_RUN_TEST(test_make_pt);
    MU_RUN_TEST(test_generalization);
    return NULL;
}

MU_RUN_SUITE(all_tests);
