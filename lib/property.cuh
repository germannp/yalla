// Container for properties that are not integrated (like cell type)
#include <string>


template<int N_MAX, typename Prop = int>
struct Property {
    Prop *h_prop = (Prop*)malloc(N_MAX*sizeof(Prop));
    Prop *d_prop;
    std::string name;
    Property (std::string init_name = "cell_type") {
        cudaMalloc(&d_prop, N_MAX*sizeof(Prop));
        name = init_name;
    }
    void memcpyHostToDevice() {
        cudaMemcpy(d_prop, h_prop, N_MAX*sizeof(Prop), cudaMemcpyHostToDevice);
    }
    void memcpyDeviceToHost() {
        cudaMemcpy(h_prop, d_prop, N_MAX*sizeof(Prop), cudaMemcpyDeviceToHost);
    }
};
