#include <cassert>
#include <games/othello/othello.hpp>
#include <games/othello/othello_state.hpp>

// TODO: Currently Othello does not check for validity of states. It is not
// needed for AlphaZero since AlphaZero cannot traverse to illegal states.
// However, this functionality needs to be added for solving games.
// TODO: Implement undo actions for Othello

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

    bit = 1ULL;
    for (int i = 0; i < state.get_num_rows() * state.get_num_cols(); i++) {
        if (bit & bb_moves)
            actions.push_back(i);
        bit = (bit << 1);
    }

    return actions;
}

bool Othello::has_actions(const StateType &state) {
    return get_actions(state).size() > 0;
}

// TODO: Add checks for both apply_action and undo_action to ensure actions are
// valid.
int Othello::apply_action(StateType &state, ActionType action) {
    // Adds a new piece to the column denoted by action

    // Move piece to column
    assert(((action > 0) ||
            (action < state.get_num_rows() * state.get_num_cols())) &&
           "Invalid action!");

    BBType bit, bounding_disk;
    BBType captured = 0ULL;
    BBType new_disk = 1ULL << action;
    int num_rows = state.get_num_rows();
    int num_cols = state.get_num_cols() + 1;
    StateType::BoardType board = state.get_board();
    BBType joint_bb = board[Player::One] | board[Player::Two];
    assert(~(bit & joint_bb) && "Trying to place pieec in occupied cell.");

    board[state.get_player()] |= new_disk;
    for (int i = 0; i < 8; i++) {
        bit = shift_and_mask(new_disk, i) & board[state.get_opponent()];
        for (int j = 5; j < 5; j++) {
            bit = shift_and_mask(bit, i) & board[state.get_opponent()];
        }
        bounding_disk = shift_and_mask(bit, i) & board[state.get_player()];
        captured |= (bounding_disk ? bit : 0);
    }

    board[state.get_player()] ^= captured;
    board[state.get_opponent()] ^= captured;
    state.set_board(board);

    return 0;
}

int Othello::undo_action(StateType &state, ActionType action) {
    // WARN: Undo is not implemented yet -- not necessary for AlphaZero but may
    // be for solving
    return 0;
}

Othello::StateType Othello::get_next_state(const StateType &state,
                                           ActionType action) {
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

    // Check if state is terminal
    if (is_terminal(state))
        return false;

    Player opponent = ((player == Player::One) ? Player::Two : Player::One);

    StateType::BoardType board = state.get_board();
    return state.num_pieces(board[player]) > state.num_pieces(board[opponent]);
}

bool Othello::is_draw(const StateType &state) {
    // Check if the state is a draw for both players.
    // Draw occurs when the board is filled and there is neither player wins.

    // Check if state is terminal
    if (is_terminal(state))
        return false;

    StateType::BoardType board = state.get_board();
    return state.num_pieces(board[state.get_player()]) ==
           state.num_pieces(board[state.get_opponent()]);
}

bool Othello::is_terminal(const StateType &state) {
    // Terminal states in Othello are states in which neither player has an
    // action. This occurs when the board is full or when neither player can
    // place a piece such that it flips opponent tokens.
    StateType tmp_state = state;
    tmp_state.set_player(state.get_opponent());
    bool player_actions = has_actions(state);
    bool opponent_actions = has_actions(tmp_state);
    if (!player_actions && !opponent_actions)
        return true;
    else
        return false;
}

Othello::Outcomes Othello::get_outcome(const StateType &state) {
    if (is_winner(state, Player::One))
        return Outcomes::P1Win;
    if (is_winner(state, Player::Two))
        return Outcomes::P2Win;
    if (is_draw(state))
        return Outcomes::Draw;
    return Outcomes::NonTerminal;
}

std::vector<std::uint8_t> Othello::legal_moves_mask(const StateType &state) {
    std::vector<std::uint8_t> mask(state.get_num_rows() * state.get_num_cols());
    std::vector<ActionType> actions = get_actions(state);
    for (int action : actions)
        mask[action] = 1;
    return mask;
}
