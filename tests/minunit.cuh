// Minimalistic testing framework ripped off "Learn C the Hard Way", that took
// it from http://www.jera.com/techinfo/jtns/jtn002.html with a dash of NumPy
#pragma once

#include <assert.h>
#include <stdio.h>


#define MU_ASSERT(message, test)     \
    do {                             \
        if (!(test)) return message; \
    } while (0)

#define MU_RUN_TEST(test)            \
    do {                             \
        auto* message = test();      \
        tests_run++;                 \
        if (message) return message; \
    } while (0)

#define MU_RUN_SUITE(suite)                   \
    int main(int argc, const char* argv[])    \
    {                                         \
        auto* result = suite();               \
        if (result != 0) {                    \
            printf("ERROR: %s\n", result);    \
        } else {                              \
            printf("ALL TESTS PASSED\n");     \
        }                                     \
        printf("Tests run: %d\n", tests_run); \
        return result != 0;                   \
    }

auto tests_run = 0;


bool isclose(float a, float b) { return fabs(a - b) <= 1e-6 + 1e-2 * fabs(b); }


template<typename Pt, int n_max, template<typename, int> class Solver>
class Solution;

template<typename Pt, int n_max, template<typename, int> class Solver>
float3 center_of_mass(Solution<Pt, n_max, Solver>& bolls)
{
    float3 com{0};
    for (auto i = 0; i < *bolls.h_n; i++) {
        com.x += bolls.h_X[i].x / *bolls.h_n;
        com.y += bolls.h_X[i].y / *bolls.h_n;
        com.z += bolls.h_X[i].z / *bolls.h_n;
    }
    return com;
}
