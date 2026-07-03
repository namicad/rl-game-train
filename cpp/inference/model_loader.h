#pragma once
#include <vector>

struct Model {
    std::vector<float> weights;
};

Model loadModel(const char* path);
float predict(const Model& m, const std::vector<float>& input);