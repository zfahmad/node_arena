#include <cassert>
#include <cstdint>
#include <games/tic_tac_toe/tic_tac_toe.hpp>
#include <games/tic_tac_toe/tic_tac_toe_state.hpp>
#include <iostream>

// TODO: Currently the game does not check for validity of states. It is not
// needed for AlphaZero since AlphaZero cannot traverse to illegal states.
// However, this functionality needs to be added for solving games.

using Player = typename TicTacToeState::Player;

constexpr int16_t WIN_MASKS[8] = {0x0007, 0x0038, 0x01C0, 0x0049,
                                  0x0092, 0x0124, 0x0111, 0x0054};

TicTacToe::TicTacToe() = default;

void TicTacToe::reset(TicTacToeState &state) {
    state.set_board({});
    state.set_player(TicTacToeState::Player::One);
}

std::vector<int> TicTacToe::get_actions(const TicTacToeState &state) const {
    std::vector<int> actions;
    int16_t joined_bb =
        state.get_board()[Player::One] | state.get_board()[Player::Two];
    for (int i = 0; i < 9; i++) {
        if (!((joined_bb >> i) & 1))
            actions.push_back(i);
    }
    return actions;
}

int TicTacToe::apply_action(TicTacToeState &state, int action) {
    assert((action < 9) && (action > 0));
    int16_t move = 1L << action;
    TicTacToeState::BoardType board = state.get_board();

    if (board[(Player::One)] | board[Player::Two] & move) {
        std::cerr << "Attempting to place piece in occupied spot." << std::endl;
        return 1;
    }
    board[state.get_player()] ^= move;
    state.set_board(board);
    return 0;
}

int TicTacToe::undo_action(TicTacToeState &state, int action) {
    assert((action < 9) && (action > 0));
    int16_t move = 1L << action;
    TicTacToeState::BoardType board = state.get_board();

    board[state.get_player()] ^= move;
    return 0;
}

TicTacToeState TicTacToe::get_next_state(const TicTacToeState &state,
                                         int action) {
    TicTacToeState next_state = state;
    apply_action(next_state, action);
    if (state.get_player() == Player::One)
        next_state.set_player(Player::Two);
    else
        next_state.set_player(Player::One);
    return next_state;
}

bool TicTacToe::is_winner(const TicTacToeState &state, Player player) {
    // Checks if the state is a win for player
    TicTacToeState::BoardType board = state.get_board();
    for (int i; i < 8; i++) {
        if ((board[player] & WIN_MASKS[i]) == WIN_MASKS[i])
            return true;
    }
    return false;
}

bool TicTacToe::is_draw(const TicTacToeState &state) {
    // Check if the state is a draw for both players
    // NOTE: This functions assumes the state was already checked for a winner
    // and will not check for a winner.
    int16_t CLEAR_MASK = 0x01FF;
    int16_t joined_bb =
        state.get_board()[Player::One] | state.get_board()[Player::Two];
    int16_t masked_board = joined_bb & CLEAR_MASK;
    if (masked_board == 511)
        return true;
    else
        return false;
}
