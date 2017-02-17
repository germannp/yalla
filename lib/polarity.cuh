// Forces for rigid single-boll-layer
#pragma once


// Calculate force from the potential U = (n_i . r_ij/r)^2/2 for points Pt
// with polarity, i.e. a unit vector n specified by -pi < Pt.phi <= pi and
// 0 <= Pt.theta <= pi.
template<typename Pt> __device__ Pt rigidity_force(Pt Xi, Pt Xj) {
    Pt dF {0};
    float3 r {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);

    float3 ni {sinf(Xi.theta)*cosf(Xi.phi), sinf(Xi.theta)*sinf(Xi.phi),
        cosf(Xi.theta)};
    auto prodi = ni.x*r.x + ni.y*r.y + ni.z*r.z;
    dF.x = - (prodi/powf(dist, 2)*ni.x - powf(prodi, 2)/powf(dist, 4)*r.x);
    dF.y = - (prodi/powf(dist, 2)*ni.y - powf(prodi, 2)/powf(dist, 4)*r.y);
    dF.z = - (prodi/powf(dist, 2)*ni.z - powf(prodi, 2)/powf(dist, 4)*r.z);

    // n1 . n2 = sin(t1)*sin(t2)*cos(p1 - p2) + cos(t1)*cos(t2)
    auto r_phi = atan2(r.y, r.x);
    auto r_theta = acosf(r.z/dist);
    dF.phi = prodi*(sinf(r_theta)*sinf(Xi.phi - r_phi));
    dF.theta = - prodi*(cosf(Xi.theta)*sinf(r_theta)*cosf(Xi.phi - r_phi) -
        sinf(Xi.theta)*cosf(r_theta));

    // Contribution from (n_j . r_ji/r)^2/2
    float3 nj {sinf(Xj.theta)*cosf(Xj.phi), sinf(Xj.theta)*sinf(Xj.phi),
        cosf(Xj.theta)};
    auto prodj = - (nj.x*r.x + nj.y*r.y + nj.z*r.z);
    dF.x += prodj/powf(dist, 2)*nj.x + powf(prodj, 2)/powf(dist, 4)*r.x;
    dF.y += prodj/powf(dist, 2)*nj.y + powf(prodj, 2)/powf(dist, 4)*r.y;
    dF.z += prodj/powf(dist, 2)*nj.z + powf(prodj, 2)/powf(dist, 4)*r.z;

    return dF;
}
