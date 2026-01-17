#include <bitset>
#include <cassert>
#include <games/connect_four/connect_four.hpp>
#include <games/connect_four/connect_four_state.hpp>
#include <iostream>

// TODO: Currently the game does not check for validity of states. It is not
// needed for AlphaZero since AlphaZero cannot traverse to illegal states.
// However, this functionality needs to be added for solving games.

using Player = typename ConnectFourState::Player;
using BBType = typename ConnectFourState::BBType;

ConnectFour::ConnectFour() = default;

void ConnectFour::reset(StateType &state) {
    state.set_board({});
    state.set_player(Player::One);
}

std::vector<ConnectFour::ActionType>
ConnectFour::get_actions(const StateType &state) const {
    // Returns a vector of columns with empty space i.e., enough space to add a
    // piece.
    std::vector<ActionType> actions;
    int num_cols = state.get_num_cols();
    BBType bit = (1L << (num_cols + 1));
    BBType joint_bb =
        state.get_board()[Player::One] | state.get_board()[Player::Two];

    for (int i = 0; i < num_cols; i++) {
        if ((joint_bb ^ bit) & bit)
            actions.push_back(i);
        bit = (bit << 1);
    }
    return actions;
}

// TODO: Add checks for both apply_action and undo_action to ensure actions are
// valid.
int ConnectFour::apply_action(StateType &state, ActionType action) {
    // Adds a new piece to the column denoted by action

    // Move piece to column
    BBType bit = 1L << action;
    int num_rows = state.get_num_rows();
    int num_cols = state.get_num_cols() + 1;
    // Move piece to bottom of column
    bit = bit << (num_cols * num_rows);
    StateType::BoardType board = state.get_board();
    BBType joint_bb = board[Player::One] | board[Player::Two];

    // Move piece up column until empty spot found
    while (bit & joint_bb)
        bit = (bit >> num_cols);

    board[state.get_player()] |= bit;
    state.set_board(board);
    return 0;
}

int ConnectFour::undo_action(StateType &state, ActionType action) {
    // Removes the top piece located in the column denoted by action
    BBType bit = 1L << action;
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

ConnectFour::StateType ConnectFour::get_next_state(const StateType &state,
                                                   ActionType action) {
    StateType next_state = state;
    apply_action(next_state, action);
    if (state.get_player() == Player::One)
        next_state.set_player(Player::Two);
    else
        next_state.set_player(Player::One);
    return next_state;
}

bool ConnectFour::shift_check(BBType board, int direction) {
    BBType CLEAR_MASK = 0UL;
    // board = board && CLEAR_MASK;
    BBType is_four = board;
    if (direction == 1) {
        for (int i = 1; i < 4; i++)
            is_four &= (board << (i * direction));
    } else {
        for (int i = 1; i < 4; i++)
            is_four &= (board >> (i * direction));
    }
    if (is_four != 0)
        return true;
    return false;
}

bool ConnectFour::is_winner(const StateType &state, Player player) {
    // Checks if the state is a win for the player passed as an argument

    // Creating array of directions for shifts
    int directions[] = {
        state.get_num_cols() + 1, // Shift up
        1,                        // Shift right
        state.get_num_cols() + 2, // Shift up-left
        state.get_num_cols()      // Shift up-right
    };

    // Checks each direction for four adjacent pieces for player
    BBType bit = 1UL;
    BBType CLEAR_MASK = 0UL;
    bit = (bit << state.get_num_cols()) - 1;
    bit = bit << (state.get_num_cols() + 1);
    for (int i = 0; i < state.get_num_rows(); i++) {
        CLEAR_MASK = CLEAR_MASK | bit;
        bit = bit << (state.get_num_cols() + 1);
    }
    std::cout << "Player: " << static_cast<int>(player) << std::endl;
    std::cout << std::bitset<64>(CLEAR_MASK) << std::endl;
    BBType board = state.get_board()[player] & CLEAR_MASK;
    for (int direction : directions) {
        std::cout << direction << std::endl;
        if (shift_check(board, direction)) {
            return true;
        }
    }

    return false;
}

bool ConnectFour::is_draw(const StateType &state) {
    // Check if the state is a draw for both players.
    // Draw occurs when the board is filled and there is neither player wins.
    // NOTE: This functions assumes the state was already checked for a winner
    // and will not check for a winner.
    BBType bit = 1UL;
    BBType CLEAR_MASK = 0UL;
    bit = (bit << state.get_num_cols()) - 1;
    bit = bit << (state.get_num_cols() + 1);
    for (int i = 0; i < state.get_num_rows(); i++) {
        bit = bit << (state.get_num_cols() + 1);
        CLEAR_MASK = CLEAR_MASK | bit;
    }

    BBType joined_bb =
        state.get_board()[Player::One] | state.get_board()[Player::Two];
    BBType masked_board = joined_bb & CLEAR_MASK;
    // std::cout << joined_bb << " " << masked_board << " " << CLEAR_MASK
    //           << std::endl;
    if (masked_board == CLEAR_MASK)
        return true;
    else
        return false;
}

bool ConnectFour::is_terminal(const StateType &state) {
    if (is_winner(state, Player::One))
        return true;
    if (is_winner(state, Player::Two))
        return true;
    if (is_draw(state))
        return true;
    return false;
}

ConnectFour::Outcomes ConnectFour::get_outcome(const StateType &state) {
    if (is_winner(state, Player::One))
        return Outcomes::P1Win;
    if (is_winner(state, Player::Two))
        return Outcomes::P2Win;
    if (is_draw(state))
        return Outcomes::Draw;
    return Outcomes::NonTerminal;
}
