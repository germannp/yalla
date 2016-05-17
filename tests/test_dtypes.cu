#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "minunit.cuh"


const char* test_float3() {
    float3 x = {1, 2, 3}, y = {5, 4, 3};

    mu_assert("ERROR: += float3 float3, x component", (x += y).x == 1 + 5);
    mu_assert("ERROR: += float3 float3, y component", (x += y).y == 2 + 4 + 4);
    mu_assert("ERROR: += float3 float3, z component", (x += y).z == 3 + 3 + 3 + 3);

    mu_assert("ERROR: *= float3 float, x component", (y *= 2).x == 5*2);
    mu_assert("ERROR: *= float3 float, y component", (y *= 2).y == 4*2*2);
    mu_assert("ERROR: *= float3 float, z component", (y *= 2).z == 3*2*2*2);

    return NULL;
}


const char* test_float4() {
    float4 x = {1, 2, 3, 4}, y = {5, 4, 3, 2};

    mu_assert("ERROR: += float4 float4, x component", (x += y).x == 1 + 5);
    mu_assert("ERROR: += float4 float4, y component", (x += y).y == 2 + 4 + 4);
    mu_assert("ERROR: += float4 float4, z component", (x += y).z == 3 + 3 + 3 + 3);
    mu_assert("ERROR: += float4 float4, w component", (x += y).w == 4 + 2 + 2 + 2 + 2);

    mu_assert("ERROR: *= float4 float, x component", (y *= 2).x == 5*2);
    mu_assert("ERROR: *= float4 float, y component", (y *= 2).y == 4*2*2);
    mu_assert("ERROR: *= float4 float, z component", (y *= 2).z == 3*2*2*2);
    mu_assert("ERROR: *= float4 float, w component", (y *= 2).w == 2*2*2*2*2);

    return NULL;
}


const char* test_generalization() {
    float3 x = {1, 2, 3}, y = {4, 3, 2};

    mu_assert("ERROR: + float3 float3, x component", (x + y).x == 5);
    mu_assert("ERROR: + float3 float3, y component", (x + y).y == 5);
    mu_assert("ERROR: + float3 float3, z component", (x + y).z == 5);

    mu_assert("ERROR: - float3 float3, x component", (x - y).x == -3);
    mu_assert("ERROR: - float3 float3, y component", (x - y).y == -1);
    mu_assert("ERROR: - float3 float3, z component", (x - y).z == 1);

    mu_assert("ERROR: * float3 float, x component", (x*3).x == 3);
    mu_assert("ERROR: * float3 float, y component", (x*3).y == 6);
    mu_assert("ERROR: * float3 float, z component", (x*3).z == 9);

    mu_assert("ERROR: * float float3, x component", (3*x).x == 3);
    mu_assert("ERROR: * float float3, y component", (3*x).y == 6);
    mu_assert("ERROR: * float float3, z component", (3*x).z == 9);

    mu_assert("ERROR: / float3 float, x component", mu_isclose((x/3).x, 1./3));
    mu_assert("ERROR: / float3 float, y component", mu_isclose((x/3).y, 2./3));
    mu_assert("ERROR: / float3 float, z component", mu_isclose((x/3).z, 1));

    mu_assert("ERROR: -= float3 float3, x component", (x -= y).x == 1 - 4);
    mu_assert("ERROR: -= float3 float3, y component", (x -= y).y == 2 - 3 - 3);
    mu_assert("ERROR: -= float3 float3, z component", (x -= y).z == 3 - 2 - 2 - 2);

    mu_assert("ERROR: /= float3 float, x component", mu_isclose((y /= 3).x, 4./3));
    mu_assert("ERROR: /= float3 float, y component", mu_isclose((y /= 3).y, 3./9));
    mu_assert("ERROR: /= float3 float, z component", mu_isclose((y /= 3).z, 2./27));

    return NULL;
}


const char* all_tests() {
    mu_run_test(test_float3);
    mu_run_test(test_float4);
    mu_run_test(test_generalization);
    return NULL;
}

mu_run_suite(all_tests)
