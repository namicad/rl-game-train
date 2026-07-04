#pragma once

#include <vector>

struct Model {
    std::vector<float> weights;
    std::vector<float> bias;
    int input_dim = 0;
    int action_dim = 0;
};

Model loadModel(const char* path);
float predict(const Model& m, const std::vector<float>& input);
std::vector<float> predict_logits(const Model& m, const std::vector<float>& input);