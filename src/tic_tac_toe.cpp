#include "tic_tac_toe.h"
#include "tic_tac_toe_state.h"
#include <cassert>
#include <cstdint>
#include <iostream>

TicTacToe::TicTacToe() = default;

void TicTacToe::reset(TicTacToeState &state) {
    state.set_board({0, 0});
    state.set_player(TicTacToeState::Player::One);
}

std::vector<int> TicTacToe::get_actions(const TicTacToeState &state) const {
    std::vector<int> actions;
    int16_t joined_bb = state.get_board()[0] | state.get_board()[1];
    for (int i = 0; i < 9; i++) {
        if (!((joined_bb >> i) & 1))
            actions.push_back(i);
    }
    return actions;
}

int TicTacToe::apply_action(TicTacToeState &state, int action) {
    assert((action < 9) && (action > 0));
    int16_t move = 1L << action;
    std::array<std::uint16_t, 2> board = state.get_board();

    if ((board[0] | board[1]) & move) {
        std::cerr << "Attempting to place piece in occupied spot." << std::endl;
        return 1;
    }
    board[static_cast<int>(state.get_player())] ^= move;
    return 0;
}

int TicTacToe::undo_action(TicTacToeState &state, int action) {
    assert((action < 9) && (action > 0));
    int16_t move = 1L << action;
    std::array<std::uint16_t, 2> board = state.get_board();

    board[static_cast<int>(state.get_player())] ^= move;
    return 0;
}
