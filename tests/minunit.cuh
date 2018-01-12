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


template<typename Pt, template<typename> class Solver>
class Solution;

template<typename Pt, template<typename> class Solver>
float3 center_of_mass(Solution<Pt, Solver>& points)
{
    float3 com{0};
    for (auto i = 0; i < *points.h_n; i++) {
        com.x += points.h_X[i].x / *points.h_n;
        com.y += points.h_X[i].y / *points.h_n;
        com.z += points.h_X[i].z / *points.h_n;
    }
    return com;
}
