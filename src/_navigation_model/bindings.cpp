#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mouse.hpp"


namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_navigation_model, m) {
    py::class_<Mouse>(m, "Mouse")
      .def(py::init<const PosInput&, double, const ActionListInput&, bool>(), "init_pos"_a, "init_ori"_a, "possible_actions"_a, "save_history"_a = true)
      .def_property_readonly("position", &Mouse::get_position)
      .def_property_readonly("initial_position", &Mouse::get_initial_position)
      .def_property_readonly("orientation", &Mouse::get_orientation)
      .def_property_readonly("initial_orientation", &Mouse::get_initial_orientation)
      .def_property_readonly("history_position", &Mouse::get_history_position)
      .def_property_readonly("history_orientation", &Mouse::get_history_orientation)
      .def("enable_history", &Mouse::enable_history, R"|(
        Toggle history saving

        Also resets history.

        :param enable: True to enable history, false otherwise
        )|",
           "enable"_a)
      .def("reset_history", &Mouse::reset_history, R"|(
        Resets history
        )|")
      .def("reset", &Mouse::reset, R"|(
        Resets position and orientation to the initial ones

        Also resets history.

        Can also change initial position and orientation.

        :param initial_pos: change initial position
        :param initial_ori: change initial orientation
        )|",
           "initial_pos"_a = py::none(), "initial_ori"_a = py::none())
      .def("get_endpoints", &Mouse::get_endpoints, R"|(
        Compute a list of possible endpoints for the mouse, with absolute coordinates

        :return: list of np.array of endpoints (2 coordinates)
        )|")
      .def("move_to", py::overload_cast<const PosInput&>(&Mouse::move_to), R"|(
        Set the new position and compute the orientation from the direction of movement

        :param new_pos: new absolute continuous position (pair of coordinates)
        :return: True if the mouse has moved, False otherwise
        )|",
           "new_pos"_a)
      .def("move_to", py::overload_cast<double, double>(&Mouse::move_to), R"|(
        Set the new position and compute the orientation from the direction of movement

        :param x: x of the new absolute continuous position
        :param y: y of the new absolute continuous position
        :return: True if the mouse has moved, False otherwise
        )|",
           "x"_a, "y"_a);
}