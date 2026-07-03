#pragma once
#include "../engine/game_state.h"
#include "../strategy/candidate.h"
#include <vector>

std::vector<float> encode(const GameState& s,
                          const Candidate& c);