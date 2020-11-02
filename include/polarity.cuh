// Forces for polarization, single-point-layer, and migration
#pragma once

#include "utils.cuh"


// We describe polarity (in and outside Pt) using a unit vector p specified
// by 0 <= p.theta < pi and -pi <= p.phi <= pi.
struct Polarity {
    float theta, phi;
};

template<typename Pt, float Pt::*theta = &Pt::theta, float Pt::*phi = &Pt::phi>
__device__ __host__ float3 pol_to_float3(Pt p)
{
    float3 vec;
    vec.x = sinf(p.*theta) * cosf(p.*phi);
    vec.y = sinf(p.*theta) * sinf(p.*phi);
    vec.z = cosf(p.*theta);
    return vec;
}

template<typename Pt>
__device__ __host__ Polarity pt_to_pol(Pt r, float dist)
{
    Polarity pol{acosf(r.z / dist), atan2(r.y, r.x)};
    return pol;
}

template<typename Pt>
__device__ __host__ Polarity pt_to_pol(Pt r)
{
#ifdef __CUDA_ARCH__
    auto dist = norm3df(r.x, r.y, r.z);
#else
    auto dist = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
#endif
    return pt_to_pol(r, dist);
}

template<typename Pt, float Pt::*theta = &Pt::theta, float Pt::*phi = &Pt::phi>
__device__ __host__ float pol_dot_product(Pt a, Polarity p)
{
    return sinf(a.*theta) * sinf(p.theta) * cosf(a.*phi - p.phi) +
           cosf(a.*theta) * cosf(p.theta);
}

// Aligning force from the potential U = - Σ(p_i . p_j), such that all
// polarities point in the same direction.
template<typename Pt, float Pt::*theta = &Pt::theta, float Pt::*phi = &Pt::phi>
__device__ __host__ Pt unidirectional_polarization_force(Pt Xi, Polarity p)
{
    Pt dF{0};
    dF.*theta = cosf(Xi.*theta) * sinf(p.theta) * cosf(Xi.*phi - p.phi) -
               sinf(Xi.*theta) * cosf(p.theta);
    auto sin_Xi_theta = sinf(Xi.*theta);
    if (fabs(sin_Xi_theta) > 1e-10)
        dF.*phi = -sinf(p.theta) * sinf(Xi.*phi - p.phi) / sin_Xi_theta;
    return dF;
}

// Aligning force from the potential U_Pol = - Σ(p_i . p_j)^2/2, such
// that all polarities are oriented the same way.
template<typename Pt, float Pt::*theta = &Pt::theta, float Pt::*phi = &Pt::phi>
__device__ __host__ Pt bidirectional_polarization_force(Pt Xi, Polarity p)
{
    auto prod = pol_dot_product<Pt, theta, phi>(Xi, p);
    return prod * unidirectional_polarization_force<Pt, theta, phi>(Xi, p);
}


// Resistance to bending from the potential U_Epi = Σ(p_i . r_ij/r)^2/2.
template<typename Pt, float Pt::*theta = &Pt::theta, float Pt::*phi = &Pt::phi>
__device__ __host__ Pt bending_force(Pt Xi, Pt r, float dist)
{
    auto pi = pol_to_float3<Pt, theta, phi>(Xi);
    auto prodi = (pi.x * r.x + pi.y * r.y + pi.z * r.z) / dist;
    auto r_hat = pt_to_pol(r, dist);
    auto dF = -prodi * unidirectional_polarization_force<Pt, theta, phi>(Xi, r_hat);

    dF.x = -prodi / dist * pi.x + powf(prodi, 2) / powf(dist, 2) * r.x;
    dF.y = -prodi / dist * pi.y + powf(prodi, 2) / powf(dist, 2) * r.y;
    dF.z = -prodi / dist * pi.z + powf(prodi, 2) / powf(dist, 2) * r.z;

    // Contribution from (p_j . r_ji/r)^2/2
    Polarity Xj{Xi.*theta - r.*theta, Xi.*phi - r.*phi};
    auto pj = pol_to_float3(Xj);
    auto prodj = (pj.x * r.x + pj.y * r.y + pj.z * r.z) / dist;
    dF.x += -prodj / dist * pj.x + powf(prodj, 2) / powf(dist, 2) * r.x;
    dF.y += -prodj / dist * pj.y + powf(prodj, 2) / powf(dist, 2) * r.y;
    dF.z += -prodj / dist * pj.z + powf(prodj, 2) / powf(dist, 2) * r.z;

    return dF;
}

// Modified bending force so that the preferential angle beteen Pi and rij
// is different from 90, thus mimicking the effect of having wedge-shaped
// epithelial cells. Note that pref_angle = Pi/2 results in a flat epithelium.
template<typename Pt>
__device__ __host__ Pt apical_constriction_force(Pt Xi, Pt r, float dist,
    float pref_angle)
{
    auto pi = pol_to_float3(Xi);
    auto prodi = (pi.x * r.x + pi.y * r.y + pi.z * r.z) / dist + cosf(pref_angle);
    auto r_hat = pt_to_pol(r, dist);
    auto dF = -prodi * unidirectional_polarization_force(Xi, r_hat);

    dF.x = -prodi / dist * pi.x + powf(prodi, 2) / powf(dist, 2) * r.x;
    dF.y = -prodi / dist * pi.y + powf(prodi, 2) / powf(dist, 2) * r.y;
    dF.z = -prodi / dist * pi.z + powf(prodi, 2) / powf(dist, 2) * r.z;

    // Contribution from (p_j . r_ji/r)^2/2
    Polarity Xj{Xi.theta - r.theta, Xi.phi - r.phi};
    auto pj = pol_to_float3(Xj);
    auto prodj = (pj.x * r.x + pj.y * r.y + pj.z * r.z) / dist - cosf(pref_angle);
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

// template<typename Pt>
template<typename Pt, float Pt::*theta = &Pt::theta, float Pt::*phi = &Pt::phi>
__device__ __host__ Pt migration_force(Pt Xi, Pt r, float dist)
{
    Pt dF{0};

    // Pulling around j
    auto r_hat = pt_to_pol(r, dist);
    if ((Xi.phi != 0) or (Xi.theta != 0)) {
        if (pol_dot_product<Pt, theta, phi>(Xi, r_hat) <= -0.15) {
            auto pi = pol_to_float3<Pt, theta, phi>(Xi);
            auto pi_T = orthonormal(r, pi);
            dF.x = 0.6 * pi.x + 0.8 * pi_T.x;
            dF.y = 0.6 * pi.y + 0.8 * pi_T.y;
            dF.z = 0.6 * pi.z + 0.8 * pi_T.z;
        }
    }

    // Getting pushed aside by j
    Polarity Xj{Xi.*theta - r.*theta, Xi.*phi - r.*phi};
    if ((Xj.phi > 1e-10) or (Xj.theta > 1e-10)) {
        if (pol_dot_product(Xj, r_hat) >= 0.15) {
            auto pj = pol_to_float3(Xj);
            auto pj_T = orthonormal(-r, pj);
            dF.x -= 0.6 * pj.x + 0.8 * pj_T.x;
            dF.y -= 0.6 * pj.y + 0.8 * pj_T.y;
            dF.z -= 0.6 * pj.z + 0.8 * pj_T.z;
        }
    }

    return dF;
}
