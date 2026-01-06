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
    std::vector<ActionType> get_actions(const TicTacToeState &state) const;
    int apply_action(TicTacToeState &state, ActionType action);
    int undo_action(TicTacToeState &state, ActionType action);
    void reset(TicTacToeState &state);
    bool is_winner(const TicTacToeState &state, StateType::Player player);
    bool is_draw(const TicTacToeState &state);
    TicTacToeState get_next_state(const TicTacToeState &state,
                                  ActionType action);
};

static_assert(Game<TicTacToe>);

#endif
