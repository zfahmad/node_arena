#include <cassert>
#include <games/othello/othello.hpp>
#include <games/othello/othello_state.hpp>
#include <iostream>

// TODO: Currently the game does not check for validity of states. It is not
// needed for AlphaZero since AlphaZero cannot traverse to illegal states.
// However, this functionality needs to be added for solving games.

using Player = typename OthelloState::Player;
using BBType = typename OthelloState::BBType;

Othello::Othello() = default;

void Othello::reset(StateType &state) {
    // Create an empty board and add initial configuration of the 4 player
    // pieces.
    BBType bb_1 = 0ULL;
    BBType bb_2 = 0ULL;
    BBType bit = 1ULL;

    // This next snippet moves to the center four spots and places pieces.
    bit = (bit << (((state.get_num_rows() / 2) - 1) * 8));
    bit = (bit << ((state.get_num_cols() / 2) - 1));
    bb_2 = bb_2 | bit;
    bit = bit << 1;
    bb_1 = bb_1 | bit;
    bit = (bit << 8);
    bb_2 = bb_2 | bit;
    bit = bit >> 1;
    bb_1 = bb_1 | bit;

    StateType::BoardType board = StateType::BoardType({bb_1, bb_2});
    state.set_board(board);
    state.set_player(Player::One);
}

BBType shift_and_mask(BBType bb, int dir) {
    // Shifts the board bb in the direction indexed by dir.
    // Then masks the bits so as to prevent wraparounds causing consecutive
    // pieces.
    // dir < 3 are board shifts to the right.
    // dir > 3 are board shifts to the left.
    if (dir > 3)
        return (bb >> SHIFTS[dir]) & SHIFT_MASKS[dir];
    else
        return (bb << SHIFTS[dir]) & SHIFT_MASKS[dir];
}

std::vector<Othello::ActionType>
Othello::get_actions(const StateType &state) const {
    // Returns a vector of columns with empty space i.e., enough space to add a
    // piece.
    std::vector<Othello::ActionType> actions;
    BBType bit = 0ULL;
    BBType bb_moves = 0ULL;
    BBType empty_cells =
        ~(state.get_board()[Player::One] | state.get_board()[Player::Two]);

    // Dumb7Fill flood algorithm to find legal moves in each of the eight
    // possible directions.
    // https://www.chessprogramming.org/Dumb7Fill
    for (int i = 0; i < 8; i++) {
        bit = shift_and_mask(state.get_board()[state.get_player()], i) &
              state.get_board()[state.get_opponent()];
        for (int j = 0; j < 5; j++) {
            bit |= shift_and_mask(bit, i) &
                   state.get_board()[state.get_opponent()];
        }
        bb_moves |= shift_and_mask(bit, i) & empty_cells;
    }

    // Mask the padded regions of the board when the dimensions are smaller than
    // 8 squares. This prevents moves adding pieces outside the board when the
    // board is small otherwise edge pieces may lead to illegal moves.
    BBType PAD_MASK = 255ULL >> (8 - state.get_num_cols());
    for (int i = 0; i < state.get_num_rows(); i++)
        PAD_MASK |= PAD_MASK << 8;
    bb_moves &= PAD_MASK;

    // StateType tmp_state = state;
    // tmp_state.set_board(StateType::BoardType({bb_moves, 0ULL}));
    // tmp_state.print_board();

    return actions;
}

// TODO: Add checks for both apply_action and undo_action to ensure actions are
// valid.
int Othello::apply_action(StateType &state, Othello::ActionType action) {
    // Adds a new piece to the column denoted by action

    // Move piece to column
    BBType bit = 1ULL << action;
    int num_rows = state.get_num_rows();
    int num_cols = state.get_num_cols() + 1;
    // Move piece to bottom of column
    // bit = bit << (num_cols * num_rows);
    // StateType::BoardType board = state.get_board();
    // BBType joint_bb = board[Player::One] | board[Player::Two];
    //
    // // Move piece up column until empty spot found
    // while (bit & joint_bb)
    //     bit = (bit >> num_cols);
    //
    // board[state.get_player()] |= bit;
    // state.set_board(board);
    return 0;
}

int Othello::undo_action(StateType &state, Othello::ActionType action) {
    // Removes the top piece located in the column denoted by action
    BBType bit = 1ULL << action;
    int num_rows = state.get_num_rows();
    int num_cols = state.get_num_cols() + 1;
    StateType::BoardType board = state.get_board();
    BBType joint_bb = board[Player::One] | board[Player::Two];

    while (bit ^ (bit & joint_bb))
        bit = (bit << num_cols);
    board[state.get_player()] ^= bit;
    state.set_board(board);
    return 0;
}

Othello::StateType Othello::get_next_state(const StateType &state,
                                           Othello::ActionType action) {
    StateType next_state = state;
    apply_action(next_state, action);
    if (state.get_player() == Player::One)
        next_state.set_player(Player::Two);
    else
        next_state.set_player(Player::One);
    return next_state;
}

bool Othello::is_winner(const StateType &state, Player player) {
    // Checks if the state is a win for the player passed as an argument

    // Creating array of directions for shifts

    // Checks each direction for four adjacent pieces for player

    return false;
}

bool Othello::is_draw(const StateType &state) {
    // Check if the state is a draw for both players.
    // Draw occurs when the board is filled and there is neither player wins.
    // NOTE: This functions assumes the state was already checked for a winner
    // and will not check for a winner.
    BBType bit = 1ULL;
    BBType CLEAR_MASK = 0ULL;
    bit = (bit << state.get_num_cols()) - 1;
    bit = bit << (state.get_num_cols() + 1);
    for (int i = 0; i < state.get_num_rows(); i++) {
        CLEAR_MASK = CLEAR_MASK | bit;
        bit = bit << (state.get_num_cols() + 1);
    }

    BBType joined_bb =
        state.get_board()[Player::One] | state.get_board()[Player::Two];
    BBType masked_board = joined_bb & CLEAR_MASK;
    std::cout << joined_bb << " " << masked_board << " " << CLEAR_MASK
              << std::endl;
    if (masked_board == CLEAR_MASK)
        return true;
    else
        return false;
}
