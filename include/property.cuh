// Container for properties that are not integrated (like cell type)
#pragma once

#include <string>


template<int n_max, typename Prop = int>
struct Property {
    Prop* h_prop = (Prop*)malloc(n_max * sizeof(Prop));
    Prop* d_prop;
    std::string name;
    Property(std::string init_name = "cell_type")
    {
        cudaMalloc(&d_prop, n_max * sizeof(Prop));
        name = init_name;
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
