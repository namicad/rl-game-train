#include "simulator.h"

void Simulator::apply(GameState& s, const Action& a)
{
    switch (a.type)
    {
        case ActionType::BUILD:
            if (s.gold >= 10)
            {
                s.gold -= 10;
                s.city_count += 1;
            }
            break;

        case ActionType::TRAIN:
            if (s.gold >= 5)
            {
                s.gold -= 5;
                s.population += 1;
            }
            break;

        case ActionType::UPGRADE:
            if (s.gold >= 20)
            {
                s.gold -= 20;
                s.population += 2;
            }
            break;

        default:
            break;
    }
}

void Simulator::enemy_step(GameState& s)
{
    // 단순 baseline enemy
    s.enemy_city_count += (s.turn % 5 == 0);
    s.population += 1;
}

void Simulator::resolve(GameState& s)
{
    s.turn++;

    s.gold += s.city_count * 2;

    if (s.city_count > s.enemy_city_count + 3)
        s.win = true;

    if (s.enemy_city_count > s.city_count + 3)
        s.lose = true;
}