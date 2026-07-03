#pragma once
#include <vector>
#include "game_state.h"

struct Observation
{
    std::vector<float> v;

    static Observation from(const GameState& s);
};