#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "minunit.cuh"

const auto n_max = 20000;
const auto dt = 0.1f;
const auto r_min = 0.8f;
const auto n_replicates = 1;
const auto threshold = 5E-4;

template<typename Pt, int n_max, template<typename, int> class Solver>
float mean_differential(
    Solution<Pt, n_max, Solver>& bolls1, Solution<Pt, n_max, Solver>& bolls2)
{
    float mean = 0.f;
    for (int i = 0 ; i < *bolls1.h_n ; i++) {
        float diff = pow(bolls1.h_X[i].x - bolls2.h_X[i].x, 2) +
            pow(bolls1.h_X[i].y - bolls2.h_X[i].y, 2) +
            pow(bolls1.h_X[i].z - bolls2.h_X[i].z, 2) ;
        mean += sqrt(diff);
        // std::cout<<"i "<<i<<" y "<<bolls1.h_X[i].y<<" "<<bolls2.h_X[i].y<<std::endl;
    }
    mean = mean/(*bolls1.h_n);
    return mean;
}

const char* test_initial_conditions()
{

    auto n_cells = 100;

    Solution<float3, n_max, Tile_solver> bolls_pre_step(n_cells);
    Solution<float3, n_max, Tile_solver> bolls_post_step(n_cells);

    //Testing sphere initial conditions
    for (int c = 1 ; c <= 7 ; c++) {
        if(c < 3)
            n_cells = pow(10, c + 1);
        else
            n_cells = 2000 * (c - 2);

        *bolls_post_step.h_n = n_cells;
        *bolls_pre_step.h_n = n_cells;
        for (int r = 1 ; r <= n_replicates ; r++) {
            uniform_sphere(0.75 * r_min, bolls_post_step);
            bolls_post_step.copy_to_host();
            for (int j = 0 ; j < n_cells ; j++)
                bolls_pre_step.h_X[j]=bolls_post_step.h_X[j];

            bolls_post_step.take_step<relaxation_linear_force,
                local_friction>(dt);

            bolls_post_step.copy_to_host();
            float diff = mean_differential(bolls_pre_step, bolls_post_step);

            MU_ASSERT("system not relaxed", diff < threshold );
        }
    }

    //Testing 2D circle initial conditions
    for (int c = 1 ; c <= 10 ; c++) {
        n_cells = 200 * c;
        *bolls_post_step.h_n = n_cells;
        *bolls_pre_step.h_n = n_cells;
        for (int r = 1 ; r <= n_replicates ; r++) {
            uniform_circle(0.75 * r_min, bolls_post_step);

            bolls_post_step.copy_to_host();
            for (int j = 0 ; j < n_cells ; j++)
                bolls_pre_step.h_X[j]=bolls_post_step.h_X[j];

            bolls_post_step.take_step<relaxation_linear_force,
                local_friction>(dt);

            bolls_post_step.copy_to_host();
            float diff = mean_differential(bolls_pre_step, bolls_post_step);

            MU_ASSERT("system not relaxed", diff < threshold );
        }
    }

    //Testing cuboid initial conditions
    for (int c = 1 ; c <= 4 ; c++) {
        float side = cbrt(500.f * c);
        float3 min_point {0};
        float3 diagonal_vector {side, side, side};
        float mean_dist = 0.8f;
        for (int r = 1 ; r <= n_replicates ; r++) {
            uniform_cuboid(mean_dist, min_point, diagonal_vector, bolls_post_step);
            *bolls_pre_step.h_n = *bolls_post_step.h_n;
            auto n_cells = *bolls_post_step.h_n;

            bolls_post_step.copy_to_host();
            for (int j = 0 ; j < n_cells ; j++)
                bolls_pre_step.h_X[j]=bolls_post_step.h_X[j];

            bolls_post_step.take_step<relaxation_linear_force,
                local_friction>(dt);

            bolls_post_step.copy_to_host();
            float diff = mean_differential(bolls_pre_step, bolls_post_step);

            MU_ASSERT("system not relaxed", diff < threshold );
        }
    }


    return NULL;
}

const char* all_tests()
{
    MU_RUN_TEST(test_initial_conditions);
    return NULL;
}

MU_RUN_SUITE(all_tests);
