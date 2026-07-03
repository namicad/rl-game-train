#include "feature.h"

std::vector<float> encode(const GameState& s,
                          const Candidate& c)
{
    std::vector<float> v;

    v.push_back(s.turn);
    v.push_back(s.resource);
    v.push_back(s.army);
    v.push_back(s.map_control);

    v.push_back(c.attack.aggression);
    v.push_back(c.attack.grouping);
    v.push_back(c.attack.target_bias);
    v.push_back(c.attack.risk);

    v.push_back(c.economy.expand);
    v.push_back(c.economy.greed);
    v.push_back(c.economy.tech);

    v.push_back(c.defense.hold);
    v.push_back(c.defense.retreat);
    v.push_back(c.defense.reinforce);

    return v;
}