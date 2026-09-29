#include <array>
#include <cassert>
#include <constants.hpp>
#include <games/lines_of_action/lines_of_action.hpp>
#include <games/lines_of_action/lines_of_action_state.hpp>
#include <iostream>
#include <bit>

// TODO: Currently the game does not check for validity of states. It is not
// needed for AlphaZero since AlphaZero cannot traverse to illegal states.
// However, this functionality needs to be added for solving games.

using Player = typename LinesOfActionState::Player;
using BBType = typename LinesOfActionState::BBType;
using BoardType = typename LinesOfActionState::BoardType;

void print_bb(std::uint64_t bb, int num_rows) {
    std::uint64_t bit = 1ULL;
    std::cout << " ";
    for (int i = 0; i < num_rows; i++) {
        std::cout << GRAY << i << " ";
    }
    std::cout << RESET << "\n";
    for (int row = 0; row < num_rows; row++) {
        std::cout << " ";
        for (int col = 0; col < num_rows; col++) {
            if (bb & bit)
                std::cout << GREEN << CROSS << " " << RESET;
            else
                std::cout << GRAY << DOT << " " << RESET;
            bit = (bit << 1);
        }
        std::cout << "\n";
        bit = (bit << (ROW_MAX - num_rows));
    }
    std::cout << std::endl;
}

LinesOfAction::LinesOfAction(int num_rows) {
    this->num_rows_ = num_rows;
    this->valid = (1 << num_rows_) - 1;
    for (int i = 0; i < num_rows_ - 1; i++)
        valid |= (valid << ROW_MAX);

    int dirs[8] = {dirs::NORTH, dirs::NORTHEAST, dirs::EAST, dirs::SOUTHEAST,
                   dirs::SOUTH, dirs::SOUTHWEST, dirs::WEST, dirs::NORTHWEST};

    // Create table of destinations and paths
    // Index 0: Direction of jump
    // Index 1: Source cell
    // Index 2: Distance of jump

    for (int k = 0; k < (ROW_MAX * ROW_MAX); k++) {
        for (int i = 0; i < ROW_MAX; i++) {
            BBType bit = 1ULL << k;
            BBType path = 0ULL;
            int shift = SHIFTS[dirs[i]];
            BBType mask = valid & SHIFT_MASKS[dirs[i]];
            dest_table[i][k][0] = 0ULL;
            path_table[i][k][0] = 0ULL;
            for (int j = 1; j < ROW_MAX; j++) {
                if (shift < 0) {
                    bit = (bit >> -shift) & mask;
                } else {
                    bit = (bit << shift) & mask;
                }
                dest_table[i][k][j] = bit;
                if (bit) {
                    path_table[i][k][j] = path;
                    path |= bit;
                } else {
                    path = 0ULL;
                    path_table[i][k][j] = path;
                }
            }
        }
    }
}

void LinesOfAction::reset(StateType &state) {
    BBType bb_1, bb_2;
    int num_rows = state.get_num_rows();

    // Set initial config for first player
    bb_1 = ((1ULL << (num_rows - 2)) - 1) << 1;
    bb_1 = bb_1 | (bb_1 << (ROW_MAX * (num_rows - 1)));

    // Set initial config for second player
    bb_2 = 0ULL;
    BBType bit = 1ULL;
    for (int i = 0; i < num_rows - 2; i++)
        bb_2 |= (bit << (i * ROW_MAX));
    bb_2 = bb_2 << ROW_MAX;
    bb_2 |= (bb_2 << (num_rows - 1));

    state.set_board(BoardType({bb_1, bb_2}));

    state.set_player(Player::One);
}

std::array<std::uint8_t, (2 * ROW_MAX) - 1>
get_for_dia_counts(BoardType board) {
    std::array<std::uint8_t, (2 * ROW_MAX) - 1> counts;
    BBType forward_diagonal = 0x0102040810204080;
    BBType loa;
    BBType joint_board = board[Player::One] | board[Player::Two];
    int ind = 0;

    for (int i = ROW_MAX - 1; i > 0; i--) {
        BBType dia = forward_diagonal >> (ROW_MAX * i);
        loa = dia & joint_board;
        counts[ind] = __builtin_popcountll(loa);
        ind++;
    }
    for (int i = 0; i < ROW_MAX; i++) {
        BBType dia = forward_diagonal << (ROW_MAX * i);
        loa = dia & joint_board;
        counts[ind] = __builtin_popcountll(loa);
        ind++;
    }

    return counts;
}

std::array<std::uint8_t, (2 * ROW_MAX) - 1>
get_back_dia_counts(BoardType board) {
    std::array<std::uint8_t, (2 * ROW_MAX) - 1> counts;
    BBType backward_diagonal = 0x8040201008040201;
    BBType loa;
    BBType joint_board = board[Player::One] | board[Player::Two];
    int ind = 0;

    for (int i = ROW_MAX - 1; i >= 0; i--) {
        BBType dia = backward_diagonal << (ROW_MAX * i);
        loa = dia & joint_board;
        counts[ind] = __builtin_popcountll(loa);
        ind++;
    }
    for (int i = 1; i < ROW_MAX; i++) {
        BBType dia = backward_diagonal >> (ROW_MAX * i);
        loa = dia & joint_board;
        counts[ind] = __builtin_popcountll(loa);
        ind++;
    }

    return counts;
}

std::array<std::uint8_t, ROW_MAX> get_hor_counts(BoardType board) {
    std::array<std::uint8_t, ROW_MAX> counts;
    BBType horizontal = 0x00000000000000FF;
    BBType loa;
    BBType joint_board = board[Player::One] | board[Player::Two];
    int ind = 0;

    for (int i = 0; i < ROW_MAX; i++) {
        BBType hor = horizontal << (ROW_MAX * i);
        loa = hor & joint_board;
        counts[ind] = __builtin_popcountll(loa);
        ind++;
    }

    return counts;
}

std::array<std::uint8_t, ROW_MAX> get_ver_counts(BoardType board) {
    std::array<std::uint8_t, ROW_MAX> counts;
    BBType vertical = 0x0101010101010101;
    BBType loa;
    BBType joint_board = board[Player::One] | board[Player::Two];
    int ind = 0;

    for (int i = 0; i < ROW_MAX; i++) {
        BBType ver = vertical << (1 * i);
        loa = ver & joint_board;
        counts[ind] = __builtin_popcountll(loa);
        ind++;
    }

    return counts;
}

std::uint8_t
get_for_dia_step(BBType bit,
                 std::array<std::uint8_t, (2 * ROW_MAX) - 1> counts) {
    std::uint8_t step;
    std::uint8_t ind = 0;
    BBType forward_diagonal = 0x0102040810204080;

    for (int i = ROW_MAX - 1; i > 0; i--) {
        BBType dia = forward_diagonal >> (ROW_MAX * i);
        if (dia & bit)
            step = counts[ind];
        ind++;
    }
    for (int i = 0; i < ROW_MAX; i++) {
        BBType dia = forward_diagonal << (ROW_MAX * i);
        if (dia & bit)
            step = counts[ind];
        ind++;
    }

    return step;
}

std::uint8_t
get_back_dia_step(BBType bit,
                  std::array<std::uint8_t, (2 * ROW_MAX) - 1> counts) {
    std::uint8_t step;
    std::uint8_t ind = 0;
    BBType backward_diagonal = 0x8040201008040201;

    for (int i = ROW_MAX - 1; i >= 0; i--) {
        BBType dia = backward_diagonal << (ROW_MAX * i);
        if (dia & bit)
            step = counts[ind];
        ind++;
    }
    for (int i = 1; i < ROW_MAX; i++) {
        BBType dia = backward_diagonal >> (ROW_MAX * i);
        if (dia & bit)
            step = counts[ind];
        ind++;
    }

    return step;
}

std::uint8_t get_hor_step(BBType bit,
                          std::array<std::uint8_t, ROW_MAX> counts) {
    std::uint8_t step;
    BBType horizontal = 0x00000000000000FF;
    std::uint8_t ind = 0;

    while (ind < ROW_MAX) {
        if (bit & horizontal) {
            step = counts[ind];
        }
        horizontal <<= ROW_MAX;
        ind++;
    }
    return step;
}

std::uint8_t get_ver_step(BBType bit,
                          std::array<std::uint8_t, ROW_MAX> counts) {
    std::uint8_t step;
    BBType vertical = 0x0101010101010101;
    std::uint8_t ind = 0;

    while (ind < ROW_MAX) {
        if (bit & vertical) {
            step = counts[ind];
        }
        vertical <<= 1;
        ind++;
    }
    return step;
}

int location_to_index(int location, int num_rows) {
    // Takes a bit location and returns its corresponding index in an array that
    // would represent the board with dimension num_rows.

    int row = (location / ROW_MAX);
    int col = location % ROW_MAX;
    int index = (row * num_rows) + col;
    return index;
}

int index_to_location(int index, int num_rows) {
    int row = index / num_rows;
    int col = index % num_rows;
    int location = (row * ROW_MAX) + col;
    return location;
}

std::array<int, 3> action_to_inds(int index, int num_rows) {
    std::array<int, 3> inds;
    int source = index / (8 * num_rows);
    int move = index % (8 * num_rows);
    int dir = move / num_rows;
    int step = move % num_rows;
    inds[0] = index_to_location(source, num_rows);
    inds[1] = dir;
    inds[2] = step;
    return inds;
}

std::vector<LinesOfAction::ActionType>
LinesOfAction::get_actions(const StateType &state) const {
    // Returns a vector of columns with empty space i.e., enough space to add a
    // piece.
    std::vector<ActionType> actions;
    BoardType board = state.get_board();
    BBType joint_board = board[Player::One] | board[Player::Two];
    BBType loa;
    int ind;

    // Count pieces on all vertical l.o.a.
    std::array<std::uint8_t, ROW_MAX> vertical_counts =
        get_ver_counts(state.get_board());

    // Count pieces on all horizontal l.o.a.
    std::array<std::uint8_t, ROW_MAX> horizontal_counts =
        get_hor_counts(state.get_board());

    // Count pieces on all forward diagonal l.o.a.
    std::array<std::uint8_t, (2 * ROW_MAX) - 1> forward_dia_counts =
        get_for_dia_counts(state.get_board());

    // Count pieces on all backward diagonal l.o.a.
    std::array<std::uint8_t, (2 * ROW_MAX) - 1> backward_dia_counts =
        get_back_dia_counts(state.get_board());

    // Create vector of positions of player's pieces
    // source is the bit position of a piece
    std::vector<std::uint8_t> sources;
    BBType bit = 1;

    BBType player_board = state.get_board()[state.get_player()];
    for (int i = 0; i < (ROW_MAX * ROW_MAX); i++) {
        if (bit & player_board)
            sources.push_back(i);
        bit <<= 1;
    }

    for (int source : sources) {
        // For a source position, retrieve possible steps available in each
        // direction
        BBType bit = 1ULL << source;
        std::uint8_t steps[8];
        int step = get_hor_step(bit, horizontal_counts);
        steps[dirs::EAST] = step;
        steps[dirs::WEST] = step;
        step = get_ver_step(bit, vertical_counts);
        steps[dirs::NORTH] = step;
        steps[dirs::SOUTH] = step;
        step = get_for_dia_step(bit, forward_dia_counts);
        steps[dirs::NORTHEAST] = step;
        steps[dirs::SOUTHWEST] = step;
        step = get_back_dia_step(bit, backward_dia_counts);
        steps[dirs::NORTHWEST] = step;
        steps[dirs::SOUTHEAST] = step;

        for (auto dir :
             {dirs::NORTH, dirs::NORTHEAST, dirs::EAST, dirs::SOUTHEAST,
              dirs::SOUTH, dirs::SOUTHWEST, dirs::WEST, dirs::NORTHWEST}) {
            BBType dest = dest_table[dir][source][steps[dir]];
            BBType path = path_table[dir][source][steps[dir]];

            // Find the bit position for the destination
            // bit_ind is -1 if the destination is not possible e.g. if the
            // destination is outside the board
            int bit_ind = dest ? __builtin_ctzll(dest) : -1;

            if (bit_ind >= 0) {
                int s = location_to_index(source, num_rows_);
                // convert the action specs to an action
                int action =
                    (s * (num_rows_ * 8)) + (dir * num_rows_) + (steps[dir]);

                // Add action to set of actions if:
                // 1. there is no opponent piece along the path of the player's
                // piece
                // 2. if there is no player piece already at the destination
                if ((!(path & board[state.get_opponent()])) &&
                    (!(dest & board[state.get_player()])))
                    actions.push_back(action);
            }
        }
    }

    return actions;
}

std::vector<LinesOfAction::ActionType>
LinesOfAction::get_reverse_actions(const StateType &state) const {
    std::vector<ActionType> actions;
    return actions;
}

// TODO: Add checks for both apply_action and undo_action to ensure actions are
// valid.
int LinesOfAction::apply_action(StateType &state, ActionType action) {
    // Get source location for move, the direction, and the step size
    std::array<int, 3> inds = action_to_inds(action, num_rows_);
    int source = inds[0];
    BBType bit = 1ULL << source;
    int dir = inds[1];
    int step = inds[2];

    // Cannot jump over opponent pieces
    // Can land on enemy pieces
    BBType path = path_table[dir][source][step];
    BBType dest = dest_table[dir][source][step];
    BoardType board = state.get_board();

    if (board[state.get_opponent()] & dest)
        board[state.get_opponent()] ^= dest;
    board[state.get_player()] ^= (bit | dest);
    state.set_board(board);

    return 0;
}

int LinesOfAction::undo_action(StateType &state, ActionType action) {
    // Removes the top piece located in the column denoted by action
    return 0;
}

LinesOfAction::StateType LinesOfAction::get_next_state(const StateType &state,
                                                       ActionType action) {
    StateType next_state = state;
    apply_action(next_state, action);
    if (state.get_player() == Player::One)
        next_state.set_player(Player::Two);
    else
        next_state.set_player(Player::One);
    return next_state;
}

LinesOfAction::StateType
LinesOfAction::get_previous_state(const StateType &state, ActionType action) {
    StateType previous_state = state;
    undo_action(previous_state, action);
    if (state.get_player() == Player::One)
        previous_state.set_player(Player::Two);
    else
        previous_state.set_player(Player::One);
    return previous_state;
}

bool LinesOfAction::is_winner(const StateType &state, Player player) {
    // Checks if the state is a win for the player passed as an argument
    BBType bit = 1ULL;
    BBType prev_bit = 0ULL;
    BBType player_bb = state.get_board()[state.get_player()];

    // Find the first bit
    while (!(bit & player_bb))
        bit <<= 1;

    while (bit != prev_bit) {
        prev_bit = bit;
        for (auto dir :
             {dirs::NORTH, dirs::NORTHEAST, dirs::EAST, dirs::SOUTHEAST,
              dirs::SOUTH, dirs::SOUTHWEST, dirs::WEST, dirs::NORTHWEST}) {
            if (SHIFTS[dir] < 0) {
                bit |= player_bb & ((bit >> -SHIFTS[dir]) & SHIFT_MASKS[dir]);
            } else {
                bit |= player_bb & ((bit << SHIFTS[dir]) & SHIFT_MASKS[dir]);
            }
        }
    }

    if (std::popcount(bit) == std::popcount(player_bb))
        return true;
    return false;
}

bool LinesOfAction::is_draw(const StateType &state) {
    // Check if the state is a draw for both players.
    // Draw occurs when the board is filled and there is neither player wins.
    if (is_winner(state, Player::One) && is_winner(state, Player::Two))
        return true;

    return false;
}

bool LinesOfAction::is_terminal(const StateType &state) {
    if (is_winner(state, Player::One))
        return true;
    if (is_winner(state, Player::Two))
        return true;
    return false;
}

LinesOfAction::Outcomes LinesOfAction::get_outcome(const StateType &state) {
    bool p1_win, p2_win;
    p1_win = is_winner(state, Player::One);
    p2_win = is_winner(state, Player::Two);
    if (p1_win && p2_win)
        return Outcomes::Draw;
    if (p1_win)
        return Outcomes::P1Win;
    if (p2_win)
        return Outcomes::P2Win;
    return Outcomes::NonTerminal;
}

std::vector<std::uint8_t>
LinesOfAction::legal_moves_mask(const StateType &state) {
    // Returns a binary vector of length num_cols.
    // A 1 represents a column where a token can be placed and 0 represents a
    // column that is full.
    std::vector<std::uint8_t> mask(num_rows_ * num_rows_ * 8 * num_rows_);
    std::vector<ActionType> actions = get_actions(state);
    for (auto action : actions)
        mask[action] = 1;
    return mask;
}

std::vector<float> LinesOfAction::decode_policy(const StateType &state,
                                                std::vector<float> policy) {
    return policy;
}
