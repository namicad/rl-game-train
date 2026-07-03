#pragma once
#include "game_state.h"
#include "action.h"
#include "observation.h"
#include "reward.h"

struct StepResult {
    Observation obs;
    float reward;
    bool done;
};

class Environment {
public:
    Environment();

    Observation reset();
    StepResult step(const Action& action);
    bool done() const;

private:
    GameState state;
    Reward reward_fn;
};