#include <games/tic_tac_toe/tic_tac_toe.hpp>
#include <games/tic_tac_toe/tic_tac_toe_state.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(tic_tac_toe_wrapper, m) {
    nb::enum_<TicTacToeState::Player>(m, "Player")
        .value("One", TicTacToeState::Player::One)
        .value("Two", TicTacToeState::Player::Two);

    nb::class_<TicTacToeState> state_class(m, "State");
    state_class
        .def(nb::init<int, int>(), nb::arg("num_rows") = 3,
             nb::arg("num_cols") = 3)
        .def("print_board", &TicTacToeState::print_board)
        .def("to_compact", &TicTacToeState::to_compact)
        .def("from_compact", &TicTacToeState::from_compact)
        .def("to_array", &TicTacToeState::to_array)
        .def("to_string", &TicTacToeState::to_string)
        .def("from_string", &TicTacToeState::from_string)
        .def("get_player", &TicTacToeState::get_player)
        .def("get_opponent", &TicTacToeState::get_opponent)
        .def("set_player", &TicTacToeState::set_player);
    state_class.attr("Player") = m.attr("Player");

    nb::enum_<TicTacToe::Outcomes>(m, "Outcomes")
        .value("NonTerminal", TicTacToe::Outcomes::NonTerminal)
        .value("P1Win", TicTacToe::Outcomes::P1Win)
        .value("P2Win", TicTacToe::Outcomes::P2Win)
        .value("Draw", TicTacToe::Outcomes::Draw);

    nb::class_<TicTacToe> game_class(m, "Game");
    game_class.def(nb::init<>())
        .def("get_id", &TicTacToe::get_id)
        .def("reset", &TicTacToe::reset)
        .def("get_actions", &TicTacToe::get_actions)
        .def("apply_action", &TicTacToe::apply_action)
        .def("get_next_state", &TicTacToe::get_next_state)
        .def("is_winner", &TicTacToe::is_winner)
        .def("is_draw", &TicTacToe::is_draw)
        .def("is_terminal", &TicTacToe::is_terminal)
        .def("get_outcome", &TicTacToe::get_outcome)
        .def("legal_moves_mask", &TicTacToe::legal_moves_mask);
    game_class.attr("Outcomes") = m.attr("Outcomes");
}
