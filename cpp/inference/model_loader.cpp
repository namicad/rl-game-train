#include "model_loader.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace {

constexpr int kInputDim = 16;
constexpr int kActionDim = 4;

}  // namespace

Model loadModel(const char* path) {
    Model m;
    const char* file_path = path != nullptr ? path : "models/weights.txt";
    std::ifstream input(file_path);
    if (!input.is_open()) {
        m.weights = {1.0f};
        m.bias = {0.0f};
        m.input_dim = 1;
        m.action_dim = 1;
        return m;
    }

    std::vector<float> values;
    std::string line;
    std::getline(input, line);
    std::istringstream iss(line);
    float value = 0.0f;
    while (iss >> value) {
        values.push_back(value);
    }

    m.input_dim = kInputDim;
    m.action_dim = kActionDim;
    const int weight_count = m.input_dim * m.action_dim;
    const int bias_count = m.action_dim;

    if (values.size() >= static_cast<size_t>(weight_count + bias_count)) {
        m.weights.assign(values.begin(), values.begin() + weight_count);
        m.bias.assign(values.begin() + weight_count, values.begin() + weight_count + bias_count);
    } else if (!values.empty()) {
        m.weights.assign(values.begin(), values.end());
        m.bias.assign(m.action_dim, 0.0f);
    } else {
        m.weights.assign(weight_count, 0.0f);
        m.bias.assign(bias_count, 0.0f);
    }

    return m;
}

std::vector<float> predict_logits(const Model& m, const std::vector<float>& input) {
    std::vector<float> logits(m.action_dim, 0.0f);
    if (m.weights.empty() || m.action_dim <= 0 || m.input_dim <= 0) {
        return logits;
    }

    for (int action = 0; action < m.action_dim; ++action) {
        float score = 0.0f;
        const int base = action * m.input_dim;
        for (int feature = 0; feature < m.input_dim; ++feature) {
            const int index = base + feature;
            if (index >= static_cast<int>(m.weights.size())) {
                break;
            }
            const float value = feature < static_cast<int>(input.size()) ? input[feature] : 0.0f;
            score += value * m.weights[index];
        }
        if (action < static_cast<int>(m.bias.size())) {
            score += m.bias[action];
        }
        logits[action] = score;
    }
    return logits;
}

float predict(const Model& m, const std::vector<float>& input) {
    const auto logits = predict_logits(m, input);
    return logits.empty() ? 0.0f : logits.front();
}