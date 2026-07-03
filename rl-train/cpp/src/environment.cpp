#include "environment.h"
#include "simulator.h"

Environment::Environment() {}

Observation Environment::reset()
{
    state = GameState(); // 초기화
    state.init();

    return Observation::from(state);
}

StepResult Environment::step(const Action& action)
{
    GameState before = state;

    Simulator::apply(state, action);
    Simulator::enemy_step(state);
    Simulator::resolve(state);

    StepResult r;
    r.reward = reward_fn(before, state);
    r.obs = Observation::from(state);
    r.done = state.is_terminal();

    return r;
}

bool Environment::done() const
{
    return state.is_terminal();
}