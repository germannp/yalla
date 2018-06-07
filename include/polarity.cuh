// Forces for polarization, single-point-layer, and migration
#pragma once

#include "utils.cuh"


// We describe polarity (in and outside Pt) using a unit vector p specified
// by 0 <= Pt.theta < pi and -pi <= Pt.phi <= pi.
struct Polarity {
    float theta, phi;
};

template<typename Pol_a, typename Pol_b>
__device__ __host__ float pol_dot_product(Pol_a a, Pol_b b)
{
    return sinf(a.theta) * sinf(b.theta) * cosf(a.phi - b.phi) +
           cosf(a.theta) * cosf(b.theta);
}


// Aligning force from the potential U = - Σ(p_i . p_j), such that all
// polarities point in the same direction.
template<typename Pt, typename Pol>
__device__ __host__ Pt unidirectional_polarization_force(Pt Xi, Pol p)
{
    Pt dF{0};
    dF.theta = cosf(Xi.theta) * sinf(p.theta) * cosf(Xi.phi - p.phi) -
               sinf(Xi.theta) * cosf(p.theta);
    auto sin_Xi_theta = sinf(Xi.theta);
    if (fabs(sin_Xi_theta) > 1e-10)
        dF.phi = -sinf(p.theta) * sinf(Xi.phi - p.phi) / sin_Xi_theta;
    return dF;
}

// Aligning force from the potential U_Pol = - Σ(p_i . p_j)^2/2, such
// that all polarities are oriented the same way.
template<typename Pt, typename Pol>
__device__ __host__ Pt bidirectional_polarization_force(Pt Xi, Pol p)
{
    auto prod = pol_dot_product(Xi, p);
    return prod * unidirectional_polarization_force(Xi, p);
}


// Resistance to bending from the potential U_Epi = Σ(p_i . r_ij/r)^2/2.
template<typename Pt>
__device__ __host__ Pt bending_force(Pt Xi, Pt r, float dist)
{
    float3 pi{sinf(Xi.theta) * cosf(Xi.phi), sinf(Xi.theta) * sinf(Xi.phi),
        cosf(Xi.theta)};
    auto prodi = (pi.x * r.x + pi.y * r.y + pi.z * r.z) / dist;
    Polarity r_hat{acosf(r.z / dist), atan2(r.y, r.x)};
    auto dF = -prodi * unidirectional_polarization_force(Xi, r_hat);

    dF.x = -prodi / dist * pi.x + powf(prodi, 2) / powf(dist, 2) * r.x;
    dF.y = -prodi / dist * pi.y + powf(prodi, 2) / powf(dist, 2) * r.y;
    dF.z = -prodi / dist * pi.z + powf(prodi, 2) / powf(dist, 2) * r.z;

    // Contribution from (p_j . r_ji/r)^2/2
    Polarity Xj{Xi.theta - r.theta, Xi.phi - r.phi};
    float3 pj{sinf(Xj.theta) * cosf(Xj.phi), sinf(Xj.theta) * sinf(Xj.phi),
        cosf(Xj.theta)};
    auto prodj = (pj.x * r.x + pj.y * r.y + pj.z * r.z) / dist;
    dF.x += -prodj / dist * pj.x + powf(prodj, 2) / powf(dist, 2) * r.x;
    dF.y += -prodj / dist * pj.y + powf(prodj, 2) / powf(dist, 2) * r.y;
    dF.z += -prodj / dist * pj.z + powf(prodj, 2) / powf(dist, 2) * r.z;

    return dF;
}


// Mono-polar migration force, after
// https://doi.org/10.1016/B978-0-12-405926-9.00016-2
template<typename Pt>
__device__ __host__ float3 orthonormal(Pt r, float3 p)
{
    float3 r3{r.x, r.y, r.z};
    auto normal = r3 - dot_product(r3, p) * p;
    return normal / sqrt(dot_product(normal, normal));
}

template<typename Pt>
__device__ __host__ Pt migration_force(Pt Xi, Pt r, float dist)
{
    Pt dF{0};

    // Pulling around j
    Polarity r_hat{acosf(r.z / dist), atan2(r.y, r.x)};
    if ((Xi.phi != 0) or (Xi.theta != 0)) {
        if (pol_dot_product(Xi, r_hat) <= -0.15) {
            float3 pi{sinf(Xi.theta) * cosf(Xi.phi),
                sinf(Xi.theta) * sinf(Xi.phi), cosf(Xi.theta)};
            auto pi_T = orthonormal(r, pi);
            dF.x = 0.6 * pi.x + 0.8 * pi_T.x;
            dF.y = 0.6 * pi.y + 0.8 * pi_T.y;
            dF.z = 0.6 * pi.z + 0.8 * pi_T.z;
        }
    }

    // Getting pushed aside by j
    Polarity Xj{Xi.theta - r.theta, Xi.phi - r.phi};
    if ((Xj.phi > 1e-10) or (Xj.theta > 1e-10)) {
        if (pol_dot_product(Xj, r_hat) >= 0.15) {
            float3 pj{sinf(Xj.theta) * cosf(Xj.phi),
                sinf(Xj.theta) * sinf(Xj.phi), cosf(Xj.theta)};
            auto pj_T = orthonormal(-r, pj);
            dF.x -= 0.6 * pj.x + 0.8 * pj_T.x;
            dF.y -= 0.6 * pj.y + 0.8 * pj_T.y;
            dF.z -= 0.6 * pj.z + 0.8 * pj_T.z;
        }
    }

    return dF;
}
