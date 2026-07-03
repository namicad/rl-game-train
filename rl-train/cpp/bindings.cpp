#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "environment.h"

namespace py = pybind11;

PYBIND11_MODULE(rl_game, m)
{
    py::class_<Observation>(m, "Observation")
        .def_readwrite("v", &Observation::v);

    py::class_<StepResult>(m, "StepResult")
        .def_readwrite("obs", &StepResult::obs)
        .def_readwrite("reward", &StepResult::reward)
        .def_readwrite("done", &StepResult::done);

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("type", &Action::type)
        .def_readwrite("unit_id", &Action::unit_id)
        .def_readwrite("target", &Action::target);

    py::enum_<ActionType>(m, "ActionType")
        .value("MOVE", ActionType::MOVE)
        .value("ATTACK", ActionType::ATTACK)
        .value("BUILD", ActionType::BUILD)
        .value("TRAIN", ActionType::TRAIN)
        .value("UPGRADE", ActionType::UPGRADE)
        .value("END", ActionType::END);

    py::class_<Environment>(m, "Environment")
        .def(py::init<>())
        .def("reset", &Environment::reset)
        .def("step", &Environment::step)
        .def("done", &Environment::done);
}