#include "mouse.hpp"

#include <pybind11/pybind11.h>

Mouse::Mouse(const PosInput& init_pos, double init_ori, const ActionListInput& possible_actions, bool save_history) {
    if (possible_actions.rows() > 0 && possible_actions.cols() != 2) {
        PyErr_SetString(PyExc_ValueError, "The list of possible actions must have shape (N, 2)");
        throw pybind11::error_already_set();
    }
    this->init_pos = init_pos;
    this->pos = init_pos;
    this->init_ori = init_ori;
    this->ori = init_ori;
    this->possible_actions = possible_actions;
    this->save_history = save_history;
    reset_history();
}

Pos Mouse::get_position() const {
    return pos;
}
Pos Mouse::get_initial_position() const {
    return init_pos;
}
double Mouse::get_orientation() const {
    return ori;
}
double Mouse::get_initial_orientation() const {
    return init_ori;
}
std::vector<Pos> Mouse::get_history_position() const {
    return history_pos;
}
std::vector<double> Mouse::get_history_orientation() const {
    return history_ori;
}

void Mouse::enable_history(bool enable) {
    save_history = enable;
    reset_history();
}

void Mouse::reset(std::optional<PosInput> initial_pos, std::optional<double> initial_ori) {
    if (initial_pos) {
        init_pos = *initial_pos;
    }
    if (initial_ori) {
        init_ori = *initial_ori;
    }
    pos = init_pos;
    ori = init_ori;
    reset_history();
}

void Mouse::reset_history() {
    history_pos.clear();
    history_ori.clear();
    if (save_history) {
        history_pos.push_back(init_pos);
        history_ori.push_back(init_ori);
    }
}

PosList Mouse::get_endpoints() const {
    Eigen::Matrix2d rot;
    rot << std::cos(ori), -std::sin(ori), std::sin(ori), std::cos(ori);

    PosList ends = PosList::Zero(possible_actions.rows(), 2);
    for (Eigen::Index i = 0; i < possible_actions.rows(); i++) {
        ends.block<1, 2>(i, 0) = rot * possible_actions.block<1, 2>(i, 0).transpose() + pos;
    }
    return ends;
}

bool Mouse::move_to(const PosInput& new_pos) {
    bool moving = false;
    if (!new_pos.isApprox(pos)) {
        ori = std::atan2(new_pos[1] - pos[1], new_pos[0] - pos[0]);
        pos = new_pos;
        moving = true;
    }
    if (save_history) {
        history_pos.push_back(pos);
        history_ori.push_back(ori);
    }
    return moving;
}

bool Mouse::move_to(double x, double y) {
    Pos p{x, y};
    return move_to(p);
}
