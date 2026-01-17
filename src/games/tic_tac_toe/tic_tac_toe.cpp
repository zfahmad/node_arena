#include <cassert>
#include <games/tic_tac_toe/tic_tac_toe.hpp>
#include <iostream>

// TODO: Currently the game does not check for validity of states. It is not
// needed for AlphaZero since AlphaZero cannot traverse to illegal states.
// However, this functionality needs to be added for solving games.

using Player = typename TicTacToe::StateType::Player;
using BBType = typename TicTacToe::StateType::BBType;

constexpr BBType WIN_MASKS[8] = {0x0007, 0x0038, 0x01C0, 0x0049,
                                 0x0092, 0x0124, 0x0111, 0x0054};

TicTacToe::TicTacToe() = default;

void TicTacToe::reset(TicTacToe::StateType &state) {
    BBType bb = 0x0000;
    state.set_board(TicTacToe::StateType::BoardType({bb, bb}));
    state.set_player(TicTacToe::StateType::Player::One);
}

std::vector<TicTacToe::ActionType>
TicTacToe::get_actions(const StateType &state) const {
    std::vector<TicTacToe::ActionType> actions;
    BBType joined_bb =
        state.get_board()[Player::One] | state.get_board()[Player::Two];
    for (int i = 0; i < 9; i++) {
        if (!((joined_bb >> i) & 1))
            actions.push_back(i);
    }
    return actions;
}

int TicTacToe::apply_action(StateType &state, ActionType action) {
    assert((action < 9) && (action > 0));
    BBType move = 1L << action;
    TicTacToe::StateType::BoardType board = state.get_board();

    if ((board[(Player::One)] | board[Player::Two]) & move) {
        std::cerr << "Attempting to place piece in occupied spot." << std::endl;
        return 1;
    }
    board[state.get_player()] ^= move;
    state.set_board(board);
    return 0;
}

int TicTacToe::undo_action(StateType &state, ActionType action) {
    assert((action < 9) && (action > 0));
    BBType move = 1L << action;
    TicTacToe::StateType::BoardType board = state.get_board();

    board[state.get_player()] ^= move;
    return 0;
}

TicTacToe::StateType TicTacToe::get_next_state(const StateType &state,
                                               ActionType action) {
    TicTacToe::StateType next_state = state;
    apply_action(next_state, action);
    if (state.get_player() == Player::One)
        next_state.set_player(Player::Two);
    else
        next_state.set_player(Player::One);
    return next_state;
}

bool TicTacToe::is_winner(const StateType &state, Player player) {
    // Checks if the state is a win for player
    TicTacToe::StateType::BoardType board = state.get_board();
    for (int i = 0; i < 8; i++) {
        if ((board[player] & WIN_MASKS[i]) == WIN_MASKS[i])
            return true;
    }
    return false;
}

bool TicTacToe::is_draw(const StateType &state) {
    // Check if the state is a draw for both players
    // NOTE: This functions assumes the state was already checked for a winner
    // and will not check for a winner.
    BBType CLEAR_MASK = 0x01FF;
    BBType joined_bb =
        state.get_board()[Player::One] | state.get_board()[Player::Two];
    BBType masked_board = joined_bb & CLEAR_MASK;
    if (masked_board == 511)
        return true;
    else
        return false;
}

bool TicTacToe::is_terminal(const StateType &state) {
    if (is_winner(state, Player::One))
        return true;
    if (is_winner(state, Player::Two))
        return true;
    if (is_draw(state))
        return true;
    return false;
}

TicTacToe::Outcomes TicTacToe::get_outcome(const StateType &state) {
    if (is_winner(state, Player::One))
        return Outcomes::P1Win;
    if (is_winner(state, Player::Two))
        return Outcomes::P2Win;
    if (is_draw(state))
        return Outcomes::Draw;
    return Outcomes::NonTerminal;
}
