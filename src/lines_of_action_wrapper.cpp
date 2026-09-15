#include <games/lines_of_action/lines_of_action.hpp>
#include <games/lines_of_action/lines_of_action_state.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/operators.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(lines_of_action_wrapper, m) {
    nb::enum_<LinesOfActionState::Player>(m, "Player")
        .value("One", LinesOfActionState::Player::One)
        .value("Two", LinesOfActionState::Player::Two);

    nb::class_<LinesOfActionState> state_class(m, "State");
    state_class
        .def(nb::init<int>(), nb::arg("num_rows") = 8)
        .def("to_array", &LinesOfActionState::to_array)
        .def("to_compact", &LinesOfActionState::to_compact)
        .def("from_compact", &LinesOfActionState::from_compact)
        .def("print_board", &LinesOfActionState::print_board)
        .def("to_string", &LinesOfActionState::to_string)
        .def("from_string", &LinesOfActionState::from_string)
        .def("get_player", &LinesOfActionState::get_player)
        .def("get_opponent", &LinesOfActionState::get_opponent)
        .def("set_player", &LinesOfActionState::set_player)
        .def(nb::self == nb::self);
    state_class.attr("Player") = m.attr("Player");

    nb::enum_<LinesOfAction::Outcomes>(m, "Outcomes")
        .value("NonTerminal", LinesOfAction::Outcomes::NonTerminal)
        .value("P1Win", LinesOfAction::Outcomes::P1Win)
        .value("P2Win", LinesOfAction::Outcomes::P2Win)
        .value("Draw", LinesOfAction::Outcomes::Draw);

    nb::class_<LinesOfAction> game_class(m, "Game");
    game_class
        .def(nb::init<int>(), nb::arg("num_rows") = 8)
        .def("get_id", &LinesOfAction::get_id)
        .def("reset", &LinesOfAction::reset)
        .def("get_actions", &LinesOfAction::get_actions)
        .def("get_reverse_actions", &LinesOfAction::get_reverse_actions)
        .def("apply_action", &LinesOfAction::apply_action)
        .def("undo_action", &LinesOfAction::undo_action)
        .def("get_next_state", &LinesOfAction::get_next_state)
        .def("get_previous_state", &LinesOfAction::get_previous_state)
        .def("is_winner", &LinesOfAction::is_winner)
        .def("is_draw", &LinesOfAction::is_draw)
        .def("is_terminal", &LinesOfAction::is_terminal)
        .def("get_outcome", &LinesOfAction::get_outcome)
        .def("legal_moves_mask", &LinesOfAction::legal_moves_mask)
        .def("decode_policy", &LinesOfAction::decode_policy);
    game_class.attr("Outcomes") = m.attr("Outcomes");
}
