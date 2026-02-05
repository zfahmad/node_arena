#include <games/othello/othello.hpp>
#include <games/othello/othello_state.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(othello_wrapper, m) {
    nb::enum_<OthelloState::Player>(m, "Player")
        .value("One", OthelloState::Player::One)
        .value("Two", OthelloState::Player::Two);

    nb::class_<OthelloState> state_class(m, "State");
    state_class
        .def(nb::init<int, int>(), nb::arg("num_rows") = 8,
             nb::arg("num_cols") = 8)
        .def("to_array", &OthelloState::to_array)
        .def("print_board", &OthelloState::print_board)
        .def("state_to_string", &OthelloState::state_to_string)
        .def("string_to_state", &OthelloState::string_to_state)
        .def("get_player", &OthelloState::get_player)
        .def("get_opponent", &OthelloState::get_opponent)
        .def("set_player", &OthelloState::set_player);
    state_class.attr("Player") = m.attr("Player");

    nb::enum_<Othello::Outcomes>(m, "Outcomes")
        .value("NonTerminal", Othello::Outcomes::NonTerminal)
        .value("P1Win", Othello::Outcomes::P1Win)
        .value("P2Win", Othello::Outcomes::P2Win)
        .value("Draw", Othello::Outcomes::Draw);

    nb::class_<Othello> game_class(m, "Game");
    game_class.def(nb::init<>())
        .def("get_id", &Othello::get_id)
        .def("reset", &Othello::reset)
        .def("get_actions", &Othello::get_actions)
        .def("apply_action", &Othello::apply_action)
        .def("get_next_state", &Othello::get_next_state)
        .def("is_winner", &Othello::is_winner)
        .def("is_draw", &Othello::is_draw)
        .def("is_terminal", &Othello::is_terminal)
        .def("get_outcome", &Othello::get_outcome)
        .def("legal_moves_mask", &Othello::legal_moves_mask);
    game_class.attr("Outcomes") = m.attr("Outcomes");
}
