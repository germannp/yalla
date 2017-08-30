#pragma once

#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>
#include "../../include/utils.cuh"

using namespace std;

class Point {
public:
    float x;
    float y;
    float z;

    Point()
    {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }

    Point(float a, float b, float c)
    {
        x = a;
        y = b;
        z = c;
    }
};

// overloaded operators
bool operator==(Point&, Point&);
bool operator!=(Point&, Point&);
Point operator+(Point, Point);
Point operator-(Point, Point);
Point operator*(Point, float);

class Ray {
public:
    Point P0;
    Point P1;
    Ray(Point a, Point b)
    {
        P0 = a;
        P1 = b;
    }
};

class Plane {
public:
    Point V0;
    Point n;

    Plane(Point a, Point b)
    {
        V0 = a;
        n = b;
    }
};

class Triangle {
public:
    Point V0;
    Point V1;
    Point V2;
    Point C;
    Point N;

    Triangle()
    {
        V0.x = 0.0f;
        V0.y = 0.0f;
        V0.z = 0.0f;
        V1.x = 0.0f;
        V1.y = 0.0f;
        V1.z = 0.0f;
        V2.x = 0.0f;
        V2.y = 0.0f;
        V2.z = 0.0f;
        C.x = 0.0f;
        C.y = 0.0f;
        C.z = 0.0f;
        N.x = 0.0f;
        N.y = 0.0f;
        N.z = 0.0f;
    }
    Triangle(Point a, Point b, Point c)
    {
        V0 = a;
        V1 = b;
        V2 = c;
        calculate_centroid();
        calculate_normal();
    }
    Triangle(Point a, Point b, Point c, Point n)
    {
        V0 = a;
        V1 = b;
        V2 = c;
        C = (V0 + V1 + V2) * (1.f / 3.f);
        N = n;
    }

    void calculate_centroid()
    {
        C = (V0 + V1 + V2) * (1.f / 3.f);
    }

    void calculate_normal()
    {
        // recalculate normal
        auto v = V2 - V0;
        auto u = V1 - V0;
        Point n(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x);
        float d = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        N = n * (1.f / d);
    }
};

// Overloaded operators for Point
bool operator==(Point& a, Point& b)
{
    if (a.x == b.x && a.y == b.y && a.z == b.z) {
        return true;
    } else {
        return false;
    }
}

bool operator!=(Point& a, Point& b)
{
    if (a.x != b.x && a.y != b.y && a.z != b.z) {
        return true;
    } else {
        return false;
    }
}

Point operator+(Point a, Point b)
{
    Point p;
    p.x = a.x + b.x;
    p.y = a.y + b.y;
    p.z = a.z + b.z;
    return p;
}

Point operator-(Point a, Point b)
{
    Point p;
    p.x = a.x - b.x;
    p.y = a.y - b.y;
    p.z = a.z - b.z;
    return p;
}

Point operator*(Point a, float s)
{
    Point p;
    p.x = a.x * s;
    p.y = a.y * s;
    p.z = a.z * s;
    return p;
}

// The following function checks if a ray intersects with a triangle
// Theory and algorithm: http://geomalgorithms.com/a06-_intersect-2.html

// Copyright 2001 softSurfer, 2012 Dan Sunday
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

// intersect3D_RayTriangle(): find the 3D intersection of a ray with a triangle
//    Input:  a ray R, and a triangle T
//    Output: *I = intersection point (when it exists)
//    Return: -1 = triangle is degenerate (a segment or point)
//             0 =  disjoint (no intersect)
//             1 =  intersect in unique point I1
//             2 =  are in the same plane

#define SMALL_NUM 0.00000001  // anything that avoids division overflow
// dot product (3D) which allows vector operations in arguments
#define dot(u, v) ((u).x * (v).x + (u).y * (v).y + (u).z * (v).z)

int intersect_3D_ray_triangle(Ray R, Triangle T, Point* I)
{
    Point u, v, n;     // triangle vectors
    Point dir, w0, w;  // ray vectors
    float r, a, b;     // params to calc ray-plane intersect

    // get triangle edge vectors and plane normal
    u = T.V1 - T.V0;
    v = T.V2 - T.V0;
    n = T.N;
    if (n.x == 0.0f && n.y == 0.0f && n.z == 0.0f)  // triangle is degenerate
        return -1;  // do not deal with this case

    dir = R.P1 - R.P0;  // ray direction vector
    w0 = R.P0 - T.V0;
    a = -dot(n, w0);
    b = dot(n, dir);
    if (fabs(b) < SMALL_NUM) {  // ray is  parallel to triangle plane
        if (a == 0)             // ray lies in triangle plane
            return 2;
        else
            return 0;  // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0)   // ray goes away from triangle
        return 0;  // => no intersect
    // for a segment, also test if (r > 1.0) => no intersect

    *I = R.P0 + (dir * r);  // intersect point of ray and plane

    // is I inside T?
    float uu, uv, vv, wu, wv, D;
    uu = dot(u, u);
    uv = dot(u, v);
    vv = dot(v, v);
    w = *I - T.V0;
    wu = dot(w, u);
    wv = dot(w, v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    float s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)  // I is outside T
        return 0;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // I is outside T
        return 0;

    return 1;  // I is in T
}


class Meix {
public:
    float surf_area;
    int n_vertices;
    int n_facets;
    std::vector<Point> vertices;
    std::vector<Triangle> facets;
    int** triangle_to_vertices;
    std::vector<vector<int> > vertex_to_triangles;

    Meix();
    Meix(std::string);
    Meix(const Meix& copy);
    Meix& operator=(const Meix& other);
    void rescale_relative(float);
    void rescale_absolute(float, bool);
    void rotate(float, float);
    void translate(Point);
    Point get_centroid();
    void test_inclusion(std::vector<Point>&, int*, Point);
    void write_vtk(std::string);

    ~Meix();
};

Meix::Meix()
{
    surf_area = 0.f;
    n_vertices = 0;
    n_facets = 0;
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
            Point P;
            P.x = stof(items[i * 3]);
            P.y = stof(items[i * 3 + 1]);
            P.z = stof(items[i * 3 + 2]);
            vertices.push_back(P);
            count++;
        }
        items.clear();
    }

    //at this point there may be a black line or not, so we have to check
    getline(input_file, line);
    items = split(line);
    if(items[0] == "POLYGONS" or items[0] == "CELLS") {
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
    std::vector<vector<int> > dummy(n_vertices, empty);
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
    std::vector<vector<int> > dummy(n_vertices, empty);
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
    std::vector<vector<int> > dummy(n_vertices, empty);
    vertex_to_triangles = dummy;
    for (int i = 0; i < n_vertices; i++)
        vertex_to_triangles[i] = other.vertex_to_triangles[i];

    return *this;
}

void Meix::rescale_relative(float resc)
{
    for (int i = 0; i < n_vertices; i++) {
        vertices[i] = vertices[i] * resc;
    }

    for (int i = 0; i < n_facets; i++) {
        facets[i].V0 = facets[i].V0 * resc;
        facets[i].V1 = facets[i].V1 * resc;
        facets[i].V2 = facets[i].V2 * resc;
        facets[i].C = facets[i].C * resc;
    }
}

void Meix::rescale_absolute(float resc, bool boundary = false)
{
    for (int i = 0; i < n_vertices; i++) {
        if (boundary && vertices[i].x == 0.f) continue;

        Point average_normal;
        for (int j = 0; j < vertex_to_triangles[i].size(); j++) {
            int triangle = vertex_to_triangles[i][j];
            average_normal = average_normal + facets[triangle].N;
        }

        float d = sqrt(pow(average_normal.x, 2) + pow(average_normal.y, 2) +
                       pow(average_normal.z, 2));
        average_normal = average_normal * (resc / d);

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
}

void Meix::translate(Point translation_vector)
{
    for (int i = 0; i < n_vertices; i++) {
        vertices[i] = vertices[i] + translation_vector;
    }

    for (int i = 0; i < n_facets; i++) {
        facets[i].V0 = facets[i].V0 + translation_vector;
        facets[i].V1 = facets[i].V1 + translation_vector;
        facets[i].V2 = facets[i].V2 + translation_vector;
        facets[i].C = facets[i].C + translation_vector;
    }
}

// Function that checks if a point is inside a closed polyhedron defined by
// a list of facets (or triangles)
void Meix::test_inclusion(
    std::vector<Point>& points, int* inclusion, Point direction)
{
    for (int i = 0; i < points.size(); i++) {
        auto p_0 = points[i];
        auto p_1 = p_0 + direction;
        Ray R(p_0, p_1);
        int intersection_count = 0;
        for (int j = 0; j < n_facets; j++) {
            auto* intersect = new Point(0.0f, 0.0f, 0.0f);
            int test = intersect_3D_ray_triangle(R, facets[j], intersect);
            if (test > 0) intersection_count++;
        }
        if (intersection_count % 2 == 0) {
            inclusion[i] = 0;
        } else {
            inclusion[i] = 1;
        }
    }
}

// writes the whole meix data structure as a vtk file
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
