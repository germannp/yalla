// Minimalistic testing framework ripped off "Learn C the Hard Way", that took
// it from http://www.jera.com/techinfo/jtns/jtn002.html with a dash of NumPy :-)
#include <stdio.h>


#define mu_isclose(a, b) fabs(a - b) <= 1e-6 + 1e-3*fabs(b)

#define mu_assert(message, test) do { if (!(test)) return message; } while (0)

#define mu_run_test(test) do { const char *message = test(); tests_run++; \
                                if (message) return message; } while (0)

#define mu_run_suite(name) int main(int argc, char **argv) { \
    const char *result = all_tests(); \
    if (result != 0) { \
        printf("%s\n", result); \
    } \
    else { \
        printf("ALL TESTS PASSED\n"); \
    } \
    printf("Tests run: %d\n", tests_run); \
    return result != 0; \
}

int tests_run = 0;
