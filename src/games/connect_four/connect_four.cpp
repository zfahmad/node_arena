#include <cassert>
#include <cstdint>
#include <games/connect_four/connect_four.hpp>
#include <games/connect_four/connect_four_state.hpp>

// TODO: Currently the game does not check for validity of states. It is not
// needed for AlphaZero since AlphaZero cannot traverse to illegal states.
// However, this functionality needs to be added for solving games.

using Player = typename ConnectFourState::Player;

ConnectFour::ConnectFour() = default;

void ConnectFour::reset(ConnectFourState &state) {
    state.set_board({});
    state.set_player(ConnectFourState::Player::One);
}

std::vector<int> ConnectFour::get_actions(const ConnectFourState &state) const {
    // Returns a vector of columns with empty space i.e., enough space to add a
    // piece.
    std::vector<int> actions;
    int num_cols = state.get_num_cols();
    uint64_t bit = (1L << num_cols);
    uint64_t joint_bb =
        state.get_board()[Player::One] | state.get_board()[Player::Two];

    for (int i = 0; i < num_cols; i++) {
        if (joint_bb ^ bit)
            actions.push_back(i);
        bit = (bit << 1);
    }
    return actions;
}

// TODO: Add checks for both apply_action and undo_action to ensure actions are
// valid.
int ConnectFour::apply_action(ConnectFourState &state, int action) {
    // Adds a new piece to the column denoted by action
    int16_t bit = 1L << action;
    int num_rows = state.get_num_rows();
    int num_cols = state.get_num_cols();
    bit = bit << (num_cols * num_rows);
    ConnectFourState::BoardType board = state.get_board();
    uint64_t joint_bb = board[Player::One] | board[Player::Two];

    while (bit & joint_bb)
        bit = (bit >> num_cols);
    board[state.get_player()] |= bit;
    state.set_board(board);
    return 0;
}

int ConnectFour::undo_action(ConnectFourState &state, int action) {
    // Removes the top piece located in the column denoted by action
    int16_t bit = 1L << action;
    int num_rows = state.get_num_rows();
    int num_cols = state.get_num_cols();
    ConnectFourState::BoardType board = state.get_board();
    uint64_t joint_bb = board[Player::One] | board[Player::Two];

    while (bit ^ (bit & joint_bb))
        bit = (bit << num_cols);
    board[state.get_player()] ^= bit;
    state.set_board(board);
    return 0;
}

ConnectFourState ConnectFour::get_next_state(const ConnectFourState &state,
                                         int action) {
    ConnectFourState next_state = state;
    apply_action(next_state, action);
    if (state.get_player() == Player::One)
        next_state.set_player(Player::Two);
    else
        next_state.set_player(Player::One);
    return next_state;
}

bool ConnectFour::is_winner(const ConnectFourState &state, Player player) {
    // Checks if the state is a win for player
    return false;
}

// bool ConnectFour::is_draw(const ConnectFourState &state) {
//     // Check if the state is a draw for both players
//     // NOTE: This functions assumes the state was already checked for a
//     winner
//     // and will not check for a winner.
//     int16_t CLEAR_MASK = 0x01FF;
//     int16_t joined_bb =
//         state.get_board()[Player::One] | state.get_board()[Player::Two];
//     int16_t masked_board = joined_bb & CLEAR_MASK;
//     if (masked_board == 511)
//         return true;
//     else
//         return false;
// }
