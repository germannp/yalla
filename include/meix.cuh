// Handle meshes for image-based models
#pragma once

#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "dtypes.cuh"
#include "utils.cuh"


struct Ray {
    float3 P0;
    float3 P1;
    Ray(float3 a, float3 b)
    {
        P0 = a;
        P1 = b;
    }
};


struct Triangle {
    float3 V0;
    float3 V1;
    float3 V2;
    float3 C;
    float3 n;
    Triangle() : Triangle(float3{0}, float3{0}, float3{0}) {}
    Triangle(float3 a, float3 b, float3 c)
    {
        V0 = a;
        V1 = b;
        V2 = c;
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


class Meix {
public:
    float surf_area;
    int n_vertices;
    int n_facets;
    float3 min_point;
    float3 diagonal_vector;
    std::vector<float3> vertices;
    std::vector<Triangle> facets;
    int** triangle_to_vertices;
    std::vector<std::vector<int>> vertex_to_triangles;
    Meix();
    Meix(std::string file_name);
    Meix(const Meix& copy);
    Meix& operator=(const Meix& other);
    void calculate_dimensions();
    void rescale_relative(float scale);
    void rescale_absolute(float scale, bool boundary);
    void translate(float3 offset);
    float3 get_centroid();
    template<typename Pt>
    bool test_exclusion(const Pt boll);
    void write_vtk(std::string);
    ~Meix();
};

Meix::Meix()
{
    surf_area = 0.f;
    n_vertices = 0;
    n_facets = 0;
    min_point = {0};
    diagonal_vector = {0};
    triangle_to_vertices = NULL;
}

Meix::Meix(std::string file_name)
{
    surf_area = 0.f;  // initialise

    std::string line;
    std::ifstream input_file;
    std::vector<std::string> items;

    input_file.open(file_name, std::fstream::in);
    assert(input_file.is_open());

    for (auto i = 0; i < 5; i++) getline(input_file, line);
    items = split(line);
    n_vertices = stoi(items[1]);
    items.clear();

    // Read vertices
    int count = 0;
    while (count < n_vertices) {
        getline(input_file, line);
        items = split(line);

        int n_points = items.size() / 3;
        for (int i = 0; i < n_points; i++) {
            float3 P;
            P.x = stof(items[i * 3]);
            P.y = stof(items[i * 3 + 1]);
            P.z = stof(items[i * 3 + 2]);
            vertices.push_back(P);
            count++;
        }
        items.clear();
    }

    // at this point there may be a black line or not, so we have to check
    getline(input_file, line);
    items = split(line);
    if (items[0] == "POLYGONS" or items[0] == "CELLS") {
        n_facets = stoi(items[1]);
        items.clear();
    } else {
        items.clear();
        getline(input_file, line);
        items = split(line);
        n_facets = stoi(items[1]);
        items.clear();
    }

    // read facets
    count = 0;

    triangle_to_vertices = (int**)malloc(n_facets * sizeof(int*));
    for (int i = 0; i < n_facets; i++)
        triangle_to_vertices[i] = (int*)malloc(3 * sizeof(int));

    while (count < n_facets) {
        getline(input_file, line);
        items = split(line);

        triangle_to_vertices[count][0] = stoi(items[1]);
        triangle_to_vertices[count][1] = stoi(items[2]);
        triangle_to_vertices[count][2] = stoi(items[3]);
        Triangle T(vertices[stoi(items[1])], vertices[stoi(items[2])],
            vertices[stoi(items[3])]);
        facets.push_back(T);
        items.clear();
        count++;
    }

    // we want to construct the list of triangles adjacent to each vertex
    // (vector of vectors)
    std::vector<int> empty;
    std::vector<std::vector<int>> dummy(n_vertices, empty);
    vertex_to_triangles = dummy;

    int vertex;
    for (int i = 0; i < n_facets; i++) {
        vertex = triangle_to_vertices[i][0];
        vertex_to_triangles[vertex].push_back(i);
        vertex = triangle_to_vertices[i][1];
        vertex_to_triangles[vertex].push_back(i);
        vertex = triangle_to_vertices[i][2];
        vertex_to_triangles[vertex].push_back(i);
    }

    calculate_dimensions();
}

Meix::Meix(const Meix& copy)
{
    surf_area = 0.f;
    n_vertices = copy.n_vertices;
    n_facets = copy.n_facets;
    vertices = copy.vertices;
    facets = copy.facets;

    triangle_to_vertices = (int**)malloc(n_facets * sizeof(int*));
    for (int i = 0; i < n_facets; i++) {
        triangle_to_vertices[i] = (int*)malloc(3 * sizeof(int));
        memcpy(triangle_to_vertices[i], copy.triangle_to_vertices[i],
            sizeof(int) * 3);
    }

    std::vector<int> empty;
    std::vector<std::vector<int>> dummy(n_vertices, empty);
    vertex_to_triangles = dummy;
    for (int i = 0; i < n_vertices; i++)
        vertex_to_triangles[i] = copy.vertex_to_triangles[i];
}

Meix& Meix::operator=(const Meix& other)
{
    surf_area = 0.f;
    n_vertices = other.n_vertices;
    n_facets = other.n_facets;
    vertices = other.vertices;
    facets = other.facets;
    triangle_to_vertices = (int**)malloc(n_facets * sizeof(int*));
    for (int i = 0; i < n_facets; i++) {
        triangle_to_vertices[i] = (int*)malloc(3 * sizeof(int));
        memcpy(triangle_to_vertices[i], other.triangle_to_vertices[i],
            sizeof(int) * 3);
    }
    std::vector<int> empty;
    std::vector<std::vector<int>> dummy(n_vertices, empty);
    vertex_to_triangles = dummy;
    for (int i = 0; i < n_vertices; i++)
        vertex_to_triangles[i] = other.vertex_to_triangles[i];

    return *this;
}

void Meix::calculate_dimensions()
{
    float3 max_point {-10000.f};
    min_point.x = 10000.f;
    min_point.y = 10000.f;
    min_point.z = 10000.f;

    for (int i = 0; i < n_vertices; i++) {
        if (vertices[i].x < min_point.x)
            min_point.x = vertices[i].x;
        if (vertices[i].x > max_point.x)
            max_point.x = vertices[i].x;
        if (vertices[i].y < min_point.y)
            min_point.y = vertices[i].y;
        if (vertices[i].y > max_point.y)
            max_point.y = vertices[i].y;
        if (vertices[i].z < min_point.z)
            min_point.z = vertices[i].z;
        if (vertices[i].z > max_point.z)
            max_point.z = vertices[i].z;
    }
    diagonal_vector.x = max_point.x - min_point.x;
    diagonal_vector.y = max_point.y - min_point.y;
    diagonal_vector.z = max_point.z - min_point.z;
}

void Meix::rescale_relative(float scale)
{
    for (int i = 0; i < n_vertices; i++) {
        vertices[i] = vertices[i] * scale;
    }

    for (int i = 0; i < n_facets; i++) {
        facets[i].V0 = facets[i].V0 * scale;
        facets[i].V1 = facets[i].V1 * scale;
        facets[i].V2 = facets[i].V2 * scale;
        facets[i].C = facets[i].C * scale;
    }

    calculate_dimensions();
}

void Meix::rescale_absolute(float scale, bool boundary = false)
{
    for (int i = 0; i < n_vertices; i++) {
        if (boundary && vertices[i].x == 0.f) continue;

        float3 average_normal;
        for (int j = 0; j < vertex_to_triangles[i].size(); j++) {
            int triangle = vertex_to_triangles[i][j];
            average_normal = average_normal + facets[triangle].n;
        }

        float d = sqrt(pow(average_normal.x, 2) + pow(average_normal.y, 2) +
                       pow(average_normal.z, 2));
        average_normal = average_normal * (scale / d);

        vertices[i] = vertices[i] + average_normal;
    }
    // rescaled the vertices, now we need to rescale the facets
    for (int i = 0; i < n_facets; i++) {
        int V0 = triangle_to_vertices[i][0];
        int V1 = triangle_to_vertices[i][1];
        int V2 = triangle_to_vertices[i][2];
        facets[i].V0 = vertices[V0];
        facets[i].V1 = vertices[V1];
        facets[i].V2 = vertices[V2];
        facets[i].calculate_centroid();
        facets[i].calculate_normal();
    }

    calculate_dimensions();
}

void Meix::translate(float3 offset)
{
    for (int i = 0; i < n_vertices; i++) {
        vertices[i] = vertices[i] + offset;
    }

    for (int i = 0; i < n_facets; i++) {
        facets[i].V0 = facets[i].V0 + offset;
        facets[i].V1 = facets[i].V1 + offset;
        facets[i].V2 = facets[i].V2 + offset;
        facets[i].C = facets[i].C + offset;
    }
}

template<typename Pt_a, typename Pt_b>
float scalar_product(Pt_a a, Pt_b b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Theory and algorithm: http://geomalgorithms.com/a06-_intersect-2.html
bool intersect(Ray R, Triangle T)
{
    // Find intersection point PI
    auto r =
        scalar_product(T.n, T.V0 - R.P0) / scalar_product(T.n, R.P1 - R.P0);
    if (r < 0) return false;  // Ray going away

    auto PI = R.P0 + ((R.P1 - R.P0) * r);

    // Check if PI in T
    auto u = T.V1 - T.V0;
    auto v = T.V2 - T.V0;
    auto w = PI - T.V0;
    auto uu = scalar_product(u, u);
    auto uv = scalar_product(u, v);
    auto vv = scalar_product(v, v);
    auto wu = scalar_product(w, u);
    auto wv = scalar_product(w, v);
    auto denom = uv * uv - uu * vv;

    auto s = (uv * wv - vv * wu) / denom;
    if (s < 0.0 or s > 1.0) return false;

    auto t = (uv * wu - uu * wv) / denom;
    if (t < 0.0 or (s + t) > 1.0) return false;

    return true;
}

template<typename Pt>
bool Meix::test_exclusion(const Pt boll)
{
    auto p_0 = float3{boll.x, boll.y, boll.z};
    auto p_1 = p_0 + float3{0.22788,  0.38849,  0.81499};
    Ray R(p_0, p_1);
    int n_intersections = 0;
    for (int j = 0; j < n_facets; j++) {
        n_intersections += intersect(R, facets[j]);
    }
    return (n_intersections % 2 == 0);
}

void Meix::write_vtk(std::string output_tag)
{
    std::string filename = "output/" + output_tag + ".meix.vtk";
    std::ofstream meix_file(filename);
    assert(meix_file.is_open());

    meix_file << "# vtk DataFile Version 3.0\n";
    meix_file << output_tag + ".meix"
              << "\n";
    meix_file << "ASCII\n";
    meix_file << "DATASET POLYDATA\n";

    meix_file << "\nPOINTS " << 3 * n_facets << " float\n";
    for (auto i = 0; i < n_facets; i++) {
        meix_file << facets[i].V0.x << " " << facets[i].V0.y << " "
                  << facets[i].V0.z << "\n";
        meix_file << facets[i].V1.x << " " << facets[i].V1.y << " "
                  << facets[i].V1.z << "\n";
        meix_file << facets[i].V2.x << " " << facets[i].V2.y << " "
                  << facets[i].V2.z << "\n";
    }

    meix_file << "\nPOLYGONS " << n_facets << " " << 4 * n_facets << "\n";
    for (auto i = 0; i < 3 * n_facets; i += 3) {
        meix_file << "3 " << i << " " << i + 1 << " " << i + 2 << "\n";
    }
    meix_file.close();
}

Meix::~Meix()
{
    vertices.clear();
    facets.clear();

    if (triangle_to_vertices != NULL) {
        for (int i = 0; i < n_facets; i++) {
            free(triangle_to_vertices[i]);
        }
        free(triangle_to_vertices);
    }

    for (int i = 0; i < vertex_to_triangles.size(); i++)
        vertex_to_triangles[i].clear();

    vertex_to_triangles.clear();
}
