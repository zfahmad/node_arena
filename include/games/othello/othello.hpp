#ifndef OTHELLO_HPP
#define OTHELLO_HPP

#include <game.hpp>
#include <games/othello/othello_state.hpp>
#include <vector>

const int SHIFTS[] = {7, 1, 9, 8, 7, 1, 9, 8};

const OthelloState::BBType SHIFT_MASKS[] = {
    0x00FEFEFEFEFEFEFE, // Up-right shift
    0xFEFEFEFEFEFEFEFE, // Right Shift
    0xFEFEFEFEFEFEFE00, // Down-right Shift
    0xFFFFFFFFFFFFFF00, // Down shift
    0x7F7F7F7F7F7F7F00, // Down-left shift
    0x7F7F7F7F7F7F7F7F, // Left shift
    0x007F7F7F7F7F7F7F, // Up-left shift
    0x00FFFFFFFFFFFFFF, // Up shift
};

class Othello {
public:
    enum class Outcomes { NonTerminal, P1Win, P2Win, Draw };
    using ActionType = int;
    using StateType = OthelloState;

    Othello();
    std::vector<ActionType> get_actions(const StateType &state) const;
    bool has_actions(const StateType &state);
    int apply_action(StateType &state, ActionType action);
    int undo_action(StateType &state, ActionType action);
    OthelloState get_next_state(const StateType &state, ActionType action);
    void reset(StateType &state);
    bool is_winner(const StateType &state, StateType::Player player);
    bool is_draw(const StateType &state);
    bool is_terminal(const StateType &state);
    Outcomes get_outcome(const StateType &state);
};

static_assert(Game<Othello>);

#endif
