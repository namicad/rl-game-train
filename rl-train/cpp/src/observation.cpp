#include "observation.h"

Observation Observation::from(const GameState& s)
{
    Observation o;

    o.v.reserve(8);

    o.v.push_back((float)s.turn);
    o.v.push_back((float)s.gold);
    o.v.push_back((float)s.population);
    o.v.push_back((float)s.city_count);
    o.v.push_back((float)s.enemy_city_count);
    o.v.push_back((float)s.kills);
    o.v.push_back((float)s.losses);

    o.v.push_back((float)(s.win ? 1 : 0));
    o.v.push_back((float)(s.lose ? 1 : 0));

    return o;
}