#include "judgment.h"
#include <vector>

using namespace std;

static vector<float> encodeState(const GameState& s) {
    return {s.turn, s.resource, s.army, s.map_control};
}

static vector<float> encodeCandidate(const Candidate& c) {
    return {
        c.attack.aggression,
        c.attack.grouping,
        c.attack.target_bias,
        c.attack.risk,

        c.economy.expand,
        c.economy.greed,
        c.economy.tech,

        c.defense.hold,
        c.defense.retreat,
        c.defense.reinforce
    };
}

static float score(const GameState& s, const Candidate& c) {
    auto vs = encodeState(s);
    auto vc = encodeCandidate(c);

    float out = 0.0f;

    for (float x : vs) out += 0.05f * x;
    for (float x : vc) out += 0.1f * x;

    return out;
}

namespace AI {

Candidate select(const GameState& state,
                 const vector<Candidate>& candidates)
{
    Candidate best = candidates[0];
    float best_score = -1e18;

    for (const auto& c : candidates) {
        float sc = score(state, c);

        if (sc > best_score) {
            best_score = sc;
            best = c;
        }
    }

    return best;
}

}