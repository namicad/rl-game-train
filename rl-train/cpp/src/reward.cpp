#include "reward.h"

float Reward::operator()(const GameState& a, const GameState& b)
{
    float r = 0;

    r += 5  * (b.gold - a.gold);
    r += 10 * (b.city_count - a.city_count);
    r += 15 * (b.population - a.population);

    r -= 10 * (b.enemy_city_count - a.enemy_city_count);

    if (b.win)  r += 1000;
    if (b.lose) r -= 1000;

    return r;
}