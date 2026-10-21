// Update inside include/engine.hpp's struct definition:
struct LinearLayerParams {
    int M;
    int N;
    int K;
    float alpha_scale;
    bool fuse_relu; // <-- Added flag
    const int8_t* d_weights;
    const float* d_bias;
};

// Update add_linear_layer signature to:
void add_linear_layer(int M, int N, int K, float alpha_scale, bool fuse_relu,
                      const int8_t* d_weights, const float* d_bias);
