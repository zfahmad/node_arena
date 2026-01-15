#include <games/connect_four/connect_four_state.hpp>
#include <games/connect_four/connect_four.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(connect_four_wrapper, m) {
    nb::class_<ConnectFourState>(m, "State")
        .def(nb::init<>())
        .def("print_board", &ConnectFourState::print_board)
        .def("state_to_string", &ConnectFourState::state_to_string)
        .def("string_to_state", &ConnectFourState::string_to_state)
        .def("get_player", &ConnectFourState::get_player)
        .def("set_player", &ConnectFourState::set_player);

    nb::enum_<ConnectFourState::Player>(m, "Player")
        .value("One", ConnectFourState::Player::One)
        .value("Two", ConnectFourState::Player::Two);

    nb::class_<ConnectFour>(m, "Game")
        .def(nb::init<>())
        .def("reset", &ConnectFour::reset)
        .def("get_actions", &ConnectFour::get_actions)
        .def("apply_action", &ConnectFour::apply_action)
        .def("get_next_state", &ConnectFour::get_next_state)
        .def("is_winner", &ConnectFour::is_winner)
        .def("is_draw", &ConnectFour::is_draw)
        .def("is_terminal", &ConnectFour::is_terminal);
}
