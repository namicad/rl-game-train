#include "model_loader.h"

Model loadModel(const char* path) {
    Model m;
    // placeholder load
    m.weights = {1.0f};
    return m;
}

float predict(const Model& m, const std::vector<float>& input) {
    float s = 0;
    for (auto v : input) s += v;
    return s;
}