#include <games/chinese_checkers/chinese_checkers.hpp>
#include <games/chinese_checkers/chinese_checkers_state.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(chinese_checkers_wrapper, m) {
    nb::enum_<ChineseCheckersState::Player>(m, "Player")
        .value("One", ChineseCheckersState::Player::One)
        .value("Two", ChineseCheckersState::Player::Two);

    nb::class_<ChineseCheckersState> state_class(m, "State");
    state_class
        .def(nb::init<int, int, int>(), nb::arg("num_rows") = 6,
             nb::arg("num_cols") = 6, nb::arg("num_pieces") = 6)
        .def("to_array", &ChineseCheckersState::to_array)
        .def("to_compact", &ChineseCheckersState::to_compact)
        .def("from_compact", &ChineseCheckersState::from_compact)
        .def("print_board", &ChineseCheckersState::print_board)
        .def("to_string", &ChineseCheckersState::to_string)
        .def("from_string", &ChineseCheckersState::from_string)
        .def("get_player", &ChineseCheckersState::get_player)
        .def("get_opponent", &ChineseCheckersState::get_opponent)
        .def("set_player", &ChineseCheckersState::set_player);
    state_class.attr("Player") = m.attr("Player");

    nb::enum_<ChineseCheckers::Outcomes>(m, "Outcomes")
        .value("NonTerminal", ChineseCheckers::Outcomes::NonTerminal)
        .value("P1Win", ChineseCheckers::Outcomes::P1Win)
        .value("P2Win", ChineseCheckers::Outcomes::P2Win)
        .value("Draw", ChineseCheckers::Outcomes::Draw);

    nb::class_<ChineseCheckers> game_class(m, "Game");
    game_class
        .def(nb::init<int, int, int>(), nb::arg("num_rows") = 6,
             nb::arg("num_cols") = 6, nb::arg("num_pieces") = 6)
        .def("get_id", &ChineseCheckers::get_id)
        .def("reset", &ChineseCheckers::reset)
        .def("get_actions", &ChineseCheckers::get_actions)
        .def("apply_action", &ChineseCheckers::apply_action)
        .def("get_next_state", &ChineseCheckers::get_next_state)
        .def("is_winner", &ChineseCheckers::is_winner)
        .def("is_draw", &ChineseCheckers::is_draw)
        .def("is_terminal", &ChineseCheckers::is_terminal)
        .def("get_outcome", &ChineseCheckers::get_outcome)
        .def("legal_moves_mask", &ChineseCheckers::legal_moves_mask)
        .def("decode_policy", &ChineseCheckers::decode_policy);
    game_class.attr("Outcomes") = m.attr("Outcomes");
}
