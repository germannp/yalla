#include "../include/dtypes.cuh"
#include "minunit.cuh"


const char* test_float3()
{
    float3 x{1, 2, 3};
    float3 y{5, 4, 3};

    MU_ASSERT("+= float3 float3, x", (x += y).x == 1 + 5);
    MU_ASSERT("+= float3 float3, y", (x += y).y == 2 + 4 + 4);
    MU_ASSERT("+= float3 float3, z", (x += y).z == 3 + 3 + 3 + 3);

    MU_ASSERT("*= float3 float, x", (y *= 2).x == 5 * 2);
    MU_ASSERT("*= float3 float, y", (y *= 2).y == 4 * 2 * 2);
    MU_ASSERT("*= float3 float, z", (y *= 2).z == 3 * 2 * 2 * 2);

    return NULL;
}


const char* test_float4()
{
    float4 x{1, 2, 3, 4};
    float4 y{5, 4, 3, 2};

    MU_ASSERT("+= float4 float4, x", (x += y).x == 1 + 5);
    MU_ASSERT("+= float4 float4, y", (x += y).y == 2 + 4 + 4);
    MU_ASSERT("+= float4 float4, z", (x += y).z == 3 + 3 + 3 + 3);
    MU_ASSERT("+= float4 float4, w", (x += y).w == 4 + 2 + 2 + 2 + 2);

    MU_ASSERT("*= float4 float, x", (y *= 2).x == 5 * 2);
    MU_ASSERT("*= float4 float, y", (y *= 2).y == 4 * 2 * 2);
    MU_ASSERT("*= float4 float, z", (y *= 2).z == 3 * 2 * 2 * 2);
    MU_ASSERT("*= float4 float, w", (y *= 2).w == 2 * 2 * 2 * 2 * 2);

    return NULL;
}


MAKE_PT(My_float4, w);

const char* test_make_pt()
{
    My_float4 x{1, 2, 3, 4};
    My_float4 y{5, 4, 3, 2};

    MU_ASSERT("+= My_float4 My_float4, x", (x += y).x == 1 + 5);
    MU_ASSERT("+= My_float4 My_float4, y", (x += y).y == 2 + 4 + 4);
    MU_ASSERT("+= My_float4 My_float4, z", (x += y).z == 3 + 3 + 3 + 3);
    MU_ASSERT("+= My_float4 My_float4, w", (x += y).w == 4 + 2 + 2 + 2 + 2);

    MU_ASSERT("*= My_float4 float, x", (y *= 2).x == 5 * 2);
    MU_ASSERT("*= My_float4 float, y", (y *= 2).y == 4 * 2 * 2);
    MU_ASSERT("*= My_float4 float, z", (y *= 2).z == 3 * 2 * 2 * 2);
    MU_ASSERT("*= My_float4 float, w", (y *= 2).w == 2 * 2 * 2 * 2 * 2);

    return NULL;
}


const char* test_generalization()
{
    float3 x{1, 2, 3};
    float3 y{4, 3, 2};

    MU_ASSERT("+ float3 float3, x", (x + y).x == 5);
    MU_ASSERT("+ float3 float3, y", (x + y).y == 5);
    MU_ASSERT("+ float3 float3, z", (x + y).z == 5);

    MU_ASSERT("- float3 float3, x", (x - y).x == -3);
    MU_ASSERT("- float3 float3, y", (x - y).y == -1);
    MU_ASSERT("- float3 float3, z", (x - y).z == 1);

    MU_ASSERT("* float3 float, x", (x * 3).x == 3);
    MU_ASSERT("* float3 float, y", (x * 3).y == 6);
    MU_ASSERT("* float3 float, z", (x * 3).z == 9);

    MU_ASSERT("* float float3, x", (3 * x).x == 3);
    MU_ASSERT("* float float3, y", (3 * x).y == 6);
    MU_ASSERT("* float float3, z", (3 * x).z == 9);

    MU_ASSERT("/ float3 float, x", isclose((x / 3).x, 1. / 3));
    MU_ASSERT("/ float3 float, y", isclose((x / 3).y, 2. / 3));
    MU_ASSERT("/ float3 float, z", isclose((x / 3).z, 1));

    MU_ASSERT("-= float3 float3, x", (x -= y).x == 1 - 4);
    MU_ASSERT("-= float3 float3, y", (x -= y).y == 2 - 3 - 3);
    MU_ASSERT("-= float3 float3, z", (x -= y).z == 3 - 2 - 2 - 2);

    MU_ASSERT("/= float3 float, x", isclose((y /= 3).x, 4. / 3));
    MU_ASSERT("/= float3 float, y", isclose((y /= 3).y, 3. / 9));
    MU_ASSERT("/= float3 float, z", isclose((y /= 3).z, 2. / 27));

    return NULL;
}


const char* all_tests()
{
    MU_RUN_TEST(test_float3);
    MU_RUN_TEST(test_float4);
    MU_RUN_TEST(test_make_pt);
    MU_RUN_TEST(test_generalization);
    return NULL;
}

MU_RUN_SUITE(all_tests);
