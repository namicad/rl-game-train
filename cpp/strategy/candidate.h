#pragma once
#include <vector>
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
};

// simple deterministic generator (MUST match RL training space)
inline vector<Candidate> generateCandidates(const GameState&) {
    return {
        { {0.9,0.8,0.2,0.7}, {0.3,0.8,0.2}, {0.7,0.2,0.6} },
        { {0.4,0.5,0.5,0.3}, {0.6,0.5,0.6}, {0.5,0.5,0.5} },
        { {0.2,0.3,0.8,0.2}, {0.9,0.2,0.7}, {0.3,0.6,0.4} },
        { {0.7,0.6,0.4,0.5}, {0.5,0.6,0.5}, {0.9,0.7,0.8} }
    };
}