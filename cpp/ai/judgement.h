#pragma once
#include "../engine/game_state.h"
#include "../strategy/candidate.h"

namespace AI {
    Candidate select(const GameState& state,
                     const vector<Candidate>& candidates);
}