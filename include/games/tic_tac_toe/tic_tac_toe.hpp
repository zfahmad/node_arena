#ifndef TIC_TAC_TOE_HPP
#define TIC_TAC_TOE_HPP

#include <game.hpp>
#include <games/tic_tac_toe/tic_tac_toe_state.hpp>
#include <vector>

class TicTacToe {
public:
    using ActionType = int;
    using StateType = TicTacToeState;

    TicTacToe();
    void reset(StateType &state);
    std::vector<ActionType> get_actions(const StateType &state) const;
    int apply_action(StateType &state, ActionType action);
    int undo_action(StateType &state, ActionType action);
    TicTacToeState get_next_state(const TicTacToeState &state,
                                  ActionType action);
    bool is_winner(const StateType &state, StateType::Player player);
    bool is_draw(const StateType &state);
    bool is_terminal(const StateType &state);
};

static_assert(Game<TicTacToe>);

#endif
