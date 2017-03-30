// Forces for rigid single-boll-layer
#pragma once


struct Polarity { float theta, phi; };

__host__ __device__ float scalar_product(float t0, float p0, float t1, float p1) {
    return sinf(t0)*sinf(t1)*cosf(p0 - p1) + cosf(t0)*cosf(t1);
}


// Calculate force from the potential U_PCP = - Σ(n_i . n_j)^2/2 for points
// Pt with polarity, i.e. a unit vector p specified by -pi <= Pt.phi <= pi
// and 0 <= Pt.theta < pi.
template<typename Pt, typename Pol> __device__ Pt pcp_force(Pt Xi, Pol pj) {
    Pt dF {0};
    auto prod = scalar_product(Xi.theta, Xi.phi, pj.theta, pj.phi);
    dF.theta += prod*(cosf(Xi.theta)*sinf(pj.theta)*cosf(Xi.phi - pj.phi) -
        sinf(Xi.theta)*cosf(pj.theta));
    auto sin_Xi_theta = sinf(Xi.theta);
    if (fabs(sin_Xi_theta) > 1e-10)
        dF.phi += - prod*sinf(pj.theta)*sinf(Xi.phi - pj.phi)/sin_Xi_theta;

    return dF;
}


// Calculate force from the potential U_Epi = Σ(p_i . r_ij/r)^2/2.
template<typename Pt> __device__ Pt rigidity_force(Pt Xi, Pt Xj) {
    Pt dF {0};
    float3 r {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    auto dist = norm3df(r.x, r.y, r.z);
    float3 pi {sinf(Xi.theta)*cosf(Xi.phi), sinf(Xi.theta)*sinf(Xi.phi),
        cosf(Xi.theta)};
    auto prodi = pi.x*r.x + pi.y*r.y + pi.z*r.z;

    auto r_theta = acosf(r.z/dist);
    auto r_phi = atan2(r.y, r.x);
    dF.theta = - prodi*(cosf(Xi.theta)*sinf(r_theta)*cosf(Xi.phi - r_phi) -
        sinf(Xi.theta)*cosf(r_theta));
    auto sin_Xi_theta = sinf(Xi.theta);
    if (fabs(sin_Xi_theta) > 1e-10)
        dF.phi = prodi*sinf(r_theta)*sinf(Xi.phi - r_phi)/sin_Xi_theta;

    dF.x = - prodi/powf(dist, 2)*pi.x + powf(prodi, 2)/powf(dist, 4)*r.x;
    dF.y = - prodi/powf(dist, 2)*pi.y + powf(prodi, 2)/powf(dist, 4)*r.y;
    dF.z = - prodi/powf(dist, 2)*pi.z + powf(prodi, 2)/powf(dist, 4)*r.z;

    // Contribution from (p_j . r_ji/r)^2/2
    float3 pj {sinf(Xj.theta)*cosf(Xj.phi), sinf(Xj.theta)*sinf(Xj.phi),
        cosf(Xj.theta)};
    auto prodj = pj.x*r.x + pj.y*r.y + pj.z*r.z;
    dF.x += - prodj/powf(dist, 2)*pj.x + powf(prodj, 2)/powf(dist, 4)*r.x;
    dF.y += - prodj/powf(dist, 2)*pj.y + powf(prodj, 2)/powf(dist, 4)*r.y;
    dF.z += - prodj/powf(dist, 2)*pj.z + powf(prodj, 2)/powf(dist, 4)*r.z;

    return dF;
}
