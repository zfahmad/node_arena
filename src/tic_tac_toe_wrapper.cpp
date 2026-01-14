#include <games/tic_tac_toe/tic_tac_toe_state.hpp>
#include <games/tic_tac_toe/tic_tac_toe.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(tic_tac_toe_wrapper, m) {
    nb::class_<TicTacToeState>(m, "State")
        .def(nb::init<>())
        .def("print_board", &TicTacToeState::print_board)
        .def("state_to_string", &TicTacToeState::state_to_string)
        .def("string_to_state", &TicTacToeState::string_to_state)
        .def("get_player", &TicTacToeState::get_player)
        .def("set_player", &TicTacToeState::set_player);

    nb::enum_<TicTacToeState::Player>(m, "Player")
        .value("One", TicTacToeState::Player::One)
        .value("Two", TicTacToeState::Player::Two);

    nb::class_<TicTacToe>(m, "Game")
        .def(nb::init<>())
        .def("reset", &TicTacToe::reset)
        .def("get_actions", &TicTacToe::get_actions)
        .def("apply_action", &TicTacToe::apply_action)
        .def("get_next_state", &TicTacToe::get_next_state)
        .def("is_winner", &TicTacToe::is_winner)
        .def("is_draw", &TicTacToe::is_draw);
}
