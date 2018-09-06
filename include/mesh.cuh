// Handle closed surface meshes for image-based models
#pragma once

#include <assert.h>
#include <math.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dtypes.cuh"
#include "solvers.cuh"
#include "utils.cuh"


// To compare the shape of a solution A with another solution or with a mesh B,
// we calculate the mean distance from each point in A to the nearest point in B
// and vice versa.

// The kernel finds the minimal distance one thread per point, to TILE_SIZE
// points at a time, after http://http.developer.nvidia.com/GPUGems3/
// gpugems3_ch31.html.
template<typename Pt1, typename Pt2>
__global__ void compute_minimum_distance(const int n1, const int n2,
    const Pt1* __restrict__ d_X1, const Pt2* __restrict__ d_X2,
    float* d_min_dist)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ Pt2 shX[TILE_SIZE];
    Pt1 Xi{0};
    if (i < n1) Xi = d_X1[i];
    float min_dist;
    for (auto tile_start = 0; tile_start < n2; tile_start += TILE_SIZE) {
        auto j = tile_start + threadIdx.x;
        if (j < n2) { shX[threadIdx.x] = d_X2[j]; }
        __syncthreads();

        for (auto k = 0; k < TILE_SIZE; k++) {
            auto j = tile_start + k;
            if ((i < n1) and (j < n2)) {
                float3 r{Xi.x - shX[k].x, Xi.y - shX[k].y, Xi.z - shX[k].z};
                auto dist = norm3df(r.x, r.y, r.z);
                if (j == 0)
                    min_dist = dist;
                else
                    min_dist = min(dist, min_dist);
            }
        }
    }
    if (i < n1) d_min_dist[i] = min_dist;
}

template<typename Pt1, typename Pt2>
float shape_comparison(const int n1, const int n2, const Pt1* __restrict__ d_X1,
    const Pt2* __restrict__ d_X2)
{
    float* d_12_dist;
    cudaMalloc(&d_12_dist, n1 * sizeof(float));
    compute_minimum_distance<<<(n1 + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
        n1, n2, d_X1, d_X2, d_12_dist);
    auto mean_12_dist =
        thrust::reduce(thrust::device, d_12_dist, d_12_dist + n1, 0.0f) / n1;
    cudaFree(d_12_dist);

    float* d_21_dist;
    cudaMalloc(&d_21_dist, n2 * sizeof(float));
    compute_minimum_distance<<<(n2 + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
        n2, n1, d_X2, d_X1, d_21_dist);
    auto mean_21_dist =
        thrust::reduce(thrust::device, d_21_dist, d_21_dist + n2, 0.0f) / n2;
    cudaFree(d_21_dist);

    return (mean_12_dist + mean_21_dist) / 2;
}

template<typename Pt1, typename Pt2, template<typename> class Solver1,
    template<typename> class Solver2>
float shape_comparison_points_to_points(
    Solution<Pt1, Solver1>& points1, Solution<Pt2, Solver2>& points2)
{
    return shape_comparison(
        points1.get_d_n(), points2.get_d_n(), points1.d_X, points2.d_X);
}


// Read, transform, and write meshes
struct Ray {
    float3 P0;
    float3 P1;
    Ray(float3 P0, float3 P1) : P0{P0}, P1{P1} {}
};

struct Triangle {
    float3 V0;
    float3 V1;
    float3 V2;
    float3 C;
    float3 n;
    Triangle() : Triangle(float3{0}, float3{0}, float3{0}) {}
    Triangle(float3 V0, float3 V1, float3 V2) : V0{V0}, V1{V1}, V2{V2}
    {
        calculate_centroid();
        calculate_normal();
    }
    void calculate_centroid() { C = (V0 + V1 + V2) / 3.f; }
    void calculate_normal()
    {
        auto v = V2 - V0;
        auto u = V1 - V0;
        n = float3{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x};
        n /= sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    }
};

class Mesh {
public:
    std::vector<float3> vertices;
    std::vector<Triangle> facets;
    float3* d_vertices;
    std::vector<std::array<int, 3>> triangle_to_vertices;
    std::vector<std::vector<int>> vertex_to_triangles;
    Mesh();
    Mesh(std::string file_name);
    Mesh(const Mesh& copy);
    ~Mesh();
    Mesh& operator=(const Mesh& other);
    float3 get_minimum();
    float3 get_maximum();
    void translate(float3 offset);
    void rotate(float around_z, float around_y, float around_x);
    void rescale(float factor);
    void grow_normally(float amount, bool boundary);
    template<typename Pt>
    bool test_exclusion(const Pt point);
    void write_vtk(std::string);
    void copy_to_device();
    template<typename Pt, template<typename> class Solver>
    float shape_comparison_mesh_to_points(Solution<Pt, Solver>& points);
};

Mesh::Mesh(std::string file_name)
{
    std::string line;
    std::ifstream input_file;
    std::vector<std::string> items;

    input_file.open(file_name, std::fstream::in);
    assert(input_file.is_open());

    auto points_start = false;
    do {
        getline(input_file, line);
        items = split(line);
        if (items.size() > 0) points_start = items[0] == "POINTS";
    } while (!points_start);
    auto n_vertices = stoi(items[1]);

    cudaMalloc(&d_vertices, n_vertices * sizeof(float3));

    // Read vertices
    auto count = 0;
    while (count < n_vertices) {
        getline(input_file, line);
        items = split(line);

        auto n_points = items.size() / 3;
        for (auto i = 0; i < n_points; i++) {
            float3 p{stof(items[i * 3]), stof(items[i * 3 + 1]),
                stof(items[i * 3 + 2])};
            vertices.push_back(p);
            count++;
        }
    }

    // Read facets
    auto polygon_start = false;
    do {
        getline(input_file, line);
        items = split(line);
        if (items.size() > 0)
            polygon_start = items[0] == "POLYGONS" or items[0] == "CELLS";
    } while (!polygon_start);
    auto n_facets = stoi(items[1]);
    assert(n_facets % 2 == 0);  // Otherwise mesh cannot be closed

    vertex_to_triangles = std::vector<std::vector<int>>(n_vertices);
    for (auto i = 0; i < n_facets; i++) {
        getline(input_file, line);
        items = split(line);
        std::array<int, 3> triangle_verts{
            stoi(items[1]), stoi(items[2]), stoi(items[3])};
        triangle_to_vertices.push_back(triangle_verts);
        Triangle T(vertices[triangle_verts[0]], vertices[triangle_verts[1]],
            vertices[triangle_verts[2]]);
        facets.push_back(T);
        vertex_to_triangles[triangle_verts[0]].push_back(i);
        vertex_to_triangles[triangle_verts[1]].push_back(i);
        vertex_to_triangles[triangle_verts[2]].push_back(i);
    }
}

Mesh::Mesh(const Mesh& copy)
{
    vertices = copy.vertices;
    facets = copy.facets;
    triangle_to_vertices = copy.triangle_to_vertices;
    vertex_to_triangles = copy.vertex_to_triangles;

    cudaMalloc(&d_vertices, vertices.size() * sizeof(float3));
}

Mesh::~Mesh() { cudaFree(d_vertices); }


float3 Mesh::get_minimum()
{
    float3 minimum{vertices[0].x, vertices[0].y, vertices[0].z};
    for (auto i = 1; i < vertices.size(); i++) {
        minimum.x = min(minimum.x, vertices[i].x);
        minimum.y = min(minimum.y, vertices[i].y);
        minimum.z = min(minimum.z, vertices[i].z);
    }
    return minimum;
}

float3 Mesh::get_maximum()
{
    float3 maximum{vertices[0].x, vertices[0].y, vertices[0].z};
    for (auto i = 1; i < vertices.size(); i++) {
        maximum.x = max(maximum.x, vertices[i].x);
        maximum.y = max(maximum.y, vertices[i].y);
        maximum.z = max(maximum.z, vertices[i].z);
    }
    return maximum;
}

void Mesh::translate(float3 offset)
{
    for (int i = 0; i < vertices.size(); i++) {
        vertices[i] = vertices[i] + offset;
    }

    for (int i = 0; i < facets.size(); i++) {
        facets[i].V0 = facets[i].V0 + offset;
        facets[i].V1 = facets[i].V1 + offset;
        facets[i].V2 = facets[i].V2 + offset;
        facets[i].C = facets[i].C + offset;
    }
}

void Mesh::rotate(float around_z, float around_y, float around_x)
{
    for (int i = 0; i < facets.size(); i++) {
        float3 old = facets[i].V0;
        facets[i].V0.x = old.x * cos(around_z) - old.y * sin(around_z);
        facets[i].V0.y = old.x * sin(around_z) + old.y * cos(around_z);

        old = facets[i].V1;
        facets[i].V1.x = old.x * cos(around_z) - old.y * sin(around_z);
        facets[i].V1.y = old.x * sin(around_z) + old.y * cos(around_z);

        old = facets[i].V2;
        facets[i].V2.x = old.x * cos(around_z) - old.y * sin(around_z);
        facets[i].V2.y = old.x * sin(around_z) + old.y * cos(around_z);

        old = facets[i].C;
        facets[i].C.x = old.x * cos(around_z) - old.y * sin(around_z);
        facets[i].C.y = old.x * sin(around_z) + old.y * cos(around_z);

        facets[i].calculate_normal();
    }
    for (int i = 0; i < vertices.size(); i++) {
        float3 old = vertices[i];
        vertices[i].x = old.x * cos(around_z) - old.y * sin(around_z);
        vertices[i].y = old.x * sin(around_z) + old.y * cos(around_z);
    }

    for (int i = 0; i < facets.size(); i++) {
        float3 old = facets[i].V0;
        facets[i].V0.x = old.x * cos(around_y) - old.z * sin(around_y);
        facets[i].V0.z = old.x * sin(around_y) + old.z * cos(around_y);

        old = facets[i].V1;
        facets[i].V1.x = old.x * cos(around_y) - old.z * sin(around_y);
        facets[i].V1.z = old.x * sin(around_y) + old.z * cos(around_y);

        old = facets[i].V2;
        facets[i].V2.x = old.x * cos(around_y) - old.z * sin(around_y);
        facets[i].V2.z = old.x * sin(around_y) + old.z * cos(around_y);

        old = facets[i].C;
        facets[i].C.x = old.x * cos(around_y) - old.z * sin(around_y);
        facets[i].C.z = old.x * sin(around_y) + old.z * cos(around_y);

        facets[i].calculate_normal();
    }
    for (int i = 0; i < vertices.size(); i++) {
        float3 old = vertices[i];
        vertices[i].x = old.x * cos(around_y) - old.z * sin(around_y);
        vertices[i].z = old.x * sin(around_y) + old.z * cos(around_y);
    }

    for (int i = 0; i < facets.size(); i++) {
        float3 old = facets[i].V0;
        facets[i].V0.y = old.y * cos(around_x) - old.z * sin(around_x);
        facets[i].V0.z = old.y * sin(around_x) + old.z * cos(around_x);

        old = facets[i].V1;
        facets[i].V1.y = old.y * cos(around_x) - old.z * sin(around_x);
        facets[i].V1.z = old.y * sin(around_x) + old.z * cos(around_x);

        old = facets[i].V2;
        facets[i].V2.y = old.y * cos(around_x) - old.z * sin(around_x);
        facets[i].V2.z = old.y * sin(around_x) + old.z * cos(around_x);

        old = facets[i].C;
        facets[i].C.y = old.y * cos(around_x) - old.z * sin(around_x);
        facets[i].C.z = old.y * sin(around_x) + old.z * cos(around_x);

        facets[i].calculate_normal();
    }
    for (int i = 0; i < vertices.size(); i++) {
        float3 old = vertices[i];
        vertices[i].y = old.y * cos(around_x) - old.z * sin(around_x);
        vertices[i].z = old.y * sin(around_x) + old.z * cos(around_x);
    }
}

void Mesh::rescale(float factor)
{
    for (int i = 0; i < vertices.size(); i++) {
        vertices[i] = vertices[i] * factor;
    }

    for (int i = 0; i < facets.size(); i++) {
        facets[i].V0 = facets[i].V0 * factor;
        facets[i].V1 = facets[i].V1 * factor;
        facets[i].V2 = facets[i].V2 * factor;
        facets[i].C = facets[i].C * factor;
    }
}

void Mesh::grow_normally(float amount, bool boundary = false)
{
    for (int i = 0; i < vertices.size(); i++) {
        if (boundary && vertices[i].x == 0.f) continue;

        float3 average_normal{0};
        for (int j = 0; j < vertex_to_triangles[i].size(); j++) {
            int triangle = vertex_to_triangles[i][j];
            average_normal = average_normal + facets[triangle].n;
        }

        float d = sqrt(pow(average_normal.x, 2) + pow(average_normal.y, 2) +
                       pow(average_normal.z, 2));
        average_normal = average_normal * (amount / d);

        vertices[i] = vertices[i] + average_normal;
    }

    for (int i = 0; i < facets.size(); i++) {
        int V0 = triangle_to_vertices[i][0];
        int V1 = triangle_to_vertices[i][1];
        int V2 = triangle_to_vertices[i][2];
        facets[i].V0 = vertices[V0];
        facets[i].V1 = vertices[V1];
        facets[i].V2 = vertices[V2];
        facets[i].calculate_centroid();
        facets[i].calculate_normal();
    }
}

// Theory and algorithm: http://geomalgorithms.com/a06-_intersect-2.html
bool intersect(Ray R, Triangle T)
{
    // Find intersection point PI
    auto r = dot_product(T.n, T.V0 - R.P0) / dot_product(T.n, R.P1 - R.P0);
    if (r < 0) return false;  // Ray going away

    auto PI = R.P0 + ((R.P1 - R.P0) * r);

    // Check if PI in T
    auto u = T.V1 - T.V0;
    auto v = T.V2 - T.V0;
    auto w = PI - T.V0;
    auto uu = dot_product(u, u);
    auto uv = dot_product(u, v);
    auto vv = dot_product(v, v);
    auto wu = dot_product(w, u);
    auto wv = dot_product(w, v);
    auto denom = uv * uv - uu * vv;

    auto s = (uv * wv - vv * wu) / denom;
    if (s < 0.0 or s > 1.0) return false;

    auto t = (uv * wu - uu * wv) / denom;
    if (t < 0.0 or (s + t) > 1.0) return false;

    return true;
}

template<typename Pt>
bool Mesh::test_exclusion(const Pt point)
{
    auto p_0 = float3{point.x, point.y, point.z};
    auto p_1 = p_0 + float3{0.22788, 0.38849, 0.81499};
    Ray R(p_0, p_1);
    int n_intersections = 0;
    for (int j = 0; j < facets.size(); j++) {
        n_intersections += intersect(R, facets[j]);
    }
    return (n_intersections % 2 == 0);
}

void Mesh::write_vtk(std::string output_tag)
{
    std::string filename = "output/" + output_tag + ".mesh.vtk";
    std::ofstream mesh_file(filename);
    assert(mesh_file.is_open());

    mesh_file << "# vtk DataFile Version 3.0\n";
    mesh_file << output_tag + ".mesh"
              << "\n";
    mesh_file << "ASCII\n";
    mesh_file << "DATASET POLYDATA\n";

    mesh_file << "\nPOINTS " << 3 * facets.size() << " float\n";
    for (auto i = 0; i < facets.size(); i++) {
        mesh_file << facets[i].V0.x << " " << facets[i].V0.y << " "
                  << facets[i].V0.z << "\n";
        mesh_file << facets[i].V1.x << " " << facets[i].V1.y << " "
                  << facets[i].V1.z << "\n";
        mesh_file << facets[i].V2.x << " " << facets[i].V2.y << " "
                  << facets[i].V2.z << "\n";
    }

    mesh_file << "\nPOLYGONS " << facets.size() << " " << 4 * facets.size()
              << "\n";
    for (auto i = 0; i < 3 * facets.size(); i += 3) {
        mesh_file << "3 " << i << " " << i + 1 << " " << i + 2 << "\n";
    }
    mesh_file.close();
}

void Mesh::copy_to_device()
{
    cudaMemcpy(d_vertices, &vertices[0], vertices.size() * sizeof(float3),
        cudaMemcpyHostToDevice);
}

template<typename Pt, template<typename> class Solver>
float Mesh::shape_comparison_mesh_to_points(Solution<Pt, Solver>& points)
{
    return shape_comparison(
        vertices.size(), points.get_d_n(), d_vertices, points.d_X);
}
