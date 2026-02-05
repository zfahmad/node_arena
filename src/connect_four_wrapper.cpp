#include <games/connect_four/connect_four.hpp>
#include <games/connect_four/connect_four_state.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(connect_four_wrapper, m) {
    nb::enum_<ConnectFourState::Player>(m, "Player")
        .value("One", ConnectFourState::Player::One)
        .value("Two", ConnectFourState::Player::Two);

    nb::class_<ConnectFourState> state_class(m, "State");
    state_class
        .def(nb::init<int, int>(), nb::arg("num_rows") = 6,
             nb::arg("num_cols") = 7)
        .def("to_array", &ConnectFourState::to_array)
        .def("print_board", &ConnectFourState::print_board)
        .def("state_to_string", &ConnectFourState::state_to_string)
        .def("string_to_state", &ConnectFourState::string_to_state)
        .def("get_player", &ConnectFourState::get_player)
        .def("get_opponent", &ConnectFourState::get_opponent)
        .def("set_player", &ConnectFourState::set_player);
    state_class.attr("Player") = m.attr("Player");

    nb::enum_<ConnectFour::Outcomes>(m, "Outcomes")
        .value("NonTerminal", ConnectFour::Outcomes::NonTerminal)
        .value("P1Win", ConnectFour::Outcomes::P1Win)
        .value("P2Win", ConnectFour::Outcomes::P2Win)
        .value("Draw", ConnectFour::Outcomes::Draw);

    nb::class_<ConnectFour> game_class(m, "Game");
    game_class.def(nb::init<>())
        .def("get_id", &ConnectFour::get_id)
        .def("reset", &ConnectFour::reset)
        .def("get_actions", &ConnectFour::get_actions)
        .def("apply_action", &ConnectFour::apply_action)
        .def("get_next_state", &ConnectFour::get_next_state)
        .def("is_winner", &ConnectFour::is_winner)
        .def("is_draw", &ConnectFour::is_draw)
        .def("is_terminal", &ConnectFour::is_terminal)
        .def("get_outcome", &ConnectFour::get_outcome)
        .def("legal_moves_mask", &ConnectFour::legal_moves_mask);
    game_class.attr("Outcomes") = m.attr("Outcomes");
}
