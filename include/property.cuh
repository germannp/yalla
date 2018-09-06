// Container for properties that are not numerically integrated (like cell type)
#pragma once

#include <string>


template<typename Prop = int>
struct Property {
    Prop* h_prop;
    Prop* d_prop;
    std::string name;
    const int n_max;
    Property(int n_max, std::string name = "cell_type")
        : n_max{n_max}, name{name}
    {
        h_prop = (Prop*)malloc(n_max * sizeof(Prop));
        cudaMalloc(&d_prop, n_max * sizeof(Prop));
    }
    ~Property()
    {
        free(h_prop);
        cudaFree(d_prop);
    }
    void copy_to_device()
    {
        cudaMemcpy(
            d_prop, h_prop, n_max * sizeof(Prop), cudaMemcpyHostToDevice);
    }
    void copy_to_host()
    {
        cudaMemcpy(
            h_prop, d_prop, n_max * sizeof(Prop), cudaMemcpyDeviceToHost);
    }
};
