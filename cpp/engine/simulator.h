#pragma once
#include "game_state.h"
#include "action.h"

class Simulator {
public:
    static void apply(GameState&, const Action&);
    static void enemy_step(GameState&);
    static void resolve(GameState&);
};