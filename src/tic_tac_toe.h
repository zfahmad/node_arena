#ifndef TIC_TAC_TOE_H
#define TIC_TAC_TOE_H

#include "../include/game.h"
#include "tic_tac_toe_state.h"
#include <vector>


class TicTacToe {
  public:
    using ActionType = int;
    using StateType = TicTacToeState;
    TicTacToe();
    std::vector<int> get_actions(const TicTacToeState &state) const;
    int apply_action(TicTacToeState &state, int action);
    int undo_action(TicTacToeState &state, int action);
    void reset(TicTacToeState &state);
    bool is_winner(TicTacToeState $state, int player);
};

static_assert(Game<TicTacToe, TicTacToeState, int>);

#endif
