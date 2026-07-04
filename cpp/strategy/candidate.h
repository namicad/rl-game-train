#pragma once
#include <vector>
#include <array>
#include <cstddef>
using namespace std;

struct AttackParams {
    float aggression;
    float grouping;
    float target_bias;
    float risk;
};

struct EconomyParams {
    float expand;
    float greed;
    float tech;
};

struct DefenseParams {
    float hold;
    float retreat;
    float reinforce;
};

struct Candidate {
    AttackParams attack;
    EconomyParams economy;
    DefenseParams defense;

    std::array<float, 12> toVector() const {
        return {
            attack.aggression,
            attack.grouping,
            attack.target_bias,
            attack.risk,
            economy.expand,
            economy.greed,
            economy.tech,
            defense.hold,
            defense.retreat,
            defense.reinforce,
            0.0f,
            0.0f
        };
    }
};

// simple deterministic generator (MUST match RL training space)
inline vector<Candidate> generateCandidates(const GameState&) {
    return {
        { {0.9f,0.8f,0.2f,0.7f}, {0.3f,0.8f,0.2f}, {0.7f,0.2f,0.6f} },
        { {0.4f,0.5f,0.5f,0.3f}, {0.6f,0.5f,0.6f}, {0.5f,0.5f,0.5f} },
        { {0.2f,0.3f,0.8f,0.2f}, {0.9f,0.2f,0.7f}, {0.3f,0.6f,0.4f} },
        { {0.7f,0.6f,0.4f,0.5f}, {0.5f,0.6f,0.5f}, {0.9f,0.7f,0.8f} }
    };
}