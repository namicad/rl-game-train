#pragma once
#include "game_state.h"

class Reward
{
public:
    float operator()(const GameState& a, const GameState& b);
};