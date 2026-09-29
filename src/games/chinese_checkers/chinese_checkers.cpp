#include <algorithm>
#include <cassert>
#include <cmath>
#include <constants.hpp>
#include <games/chinese_checkers/chinese_checkers.hpp>
#include <games/chinese_checkers/chinese_checkers_state.hpp>
#include <iostream>

// TODO: Currently ChineseCheckers does not check for validity of states. It is
// not needed for AlphaZero since AlphaZero cannot traverse to illegal states.
// However, this functionality needs to be added for solving games.
// TODO: Implement undo actions for ChineseCheckers

using Player = typename ChineseCheckersState::Player;
using BBType = typename ChineseCheckersState::BBType;

BBType generate_destinations_mask(int num_rows, int num_cols) {
    // Masks out non-valid destinations for a given board size.
    BBType mask = ~(uint128_t)0;
    BBType empty = ((uint128_t)1 << num_cols) - 1;
    empty = empty << (MAX_ROW + 1);
    for (int i = 0; i < num_rows; i++) {
        mask = mask ^ empty;
        empty = empty << MAX_ROW;
    }
    return mask;
}

BBType generate_initial_configuration(int num_pieces) {
    BBType initial = (uint128_t)0;
    int i = 0;

    do {
        initial += (uint128_t)1 << STARTING_LOCATIONS[num_pieces - 1][i];
        i++;
    } while ((STARTING_LOCATIONS[num_pieces - 1][i] != -1) && (i < 11));

    return initial;
}

std::vector<std::vector<int>> get_initial_piece_locs(int num_rows,
                                                     int num_pieces) {
    std::vector<std::vector<int>> locations;

    std::vector<int> player_locations;
    int i = 0;
    do {
        player_locations.push_back(STARTING_LOCATIONS[num_pieces - 1][i]);
        i++;
    } while ((STARTING_LOCATIONS[num_pieces - 1][i] != -1) && (i < 10));
    locations.push_back(player_locations);
    player_locations.clear();

    i = 0;
    int offset = (MAX_ROW + 1) - num_rows;
    int final_bit = 108 - (offset * 122);

    do {
        player_locations.push_back(final_bit -
                                   STARTING_LOCATIONS[num_pieces - 1][i]);
        i++;
    } while ((STARTING_LOCATIONS[num_pieces - 1][i] != -1) && (i < 10));
    locations.push_back(player_locations);

    return locations;
}

BBType generate_empties_mask(int num_rows, int num_cols, int num_pieces) {
    // Masks out non-valid empty locations for a given size.
    // Depending on the dimensions and the number of pieces, players may use
    // other goal regions for intermediary hops.
    BBType mask = ~(uint128_t)0;
    BBType empty = ((uint128_t)1 << num_cols) - 1;
    empty = empty << 12;
    for (int i = 0; i < num_rows; i++) {
        mask = mask ^ empty;
        empty = empty << 11;
    }

    // Check if board size and number of pieces allow for empty goal spaces.
    if ((num_rows == 5) && (num_pieces < 4)) {
        BBType outside = ((uint128_t)1 << 4) + ((uint128_t)1 << 5) +
                         ((uint128_t)1 << 17) + ((uint128_t)1 << 28) +
                         ((uint128_t)1 << 44) + ((uint128_t)1 << 55) +
                         ((uint128_t)1 << 67) + ((uint128_t)1 << 68);
        mask = mask ^ outside;
    }

    if ((num_rows == 7) && (num_pieces < 7)) {
        BBType outside = ((uint128_t)1 << 5) + ((uint128_t)1 << 6) + ((uint128_t)1 << 7) +
                         ((uint128_t)1 << 19) + ((uint128_t)1 << 30) + ((uint128_t)1 << 41) +
                         ((uint128_t)1 << 55) + ((uint128_t)1 << 66) + ((uint128_t)1 << 77) +
                         ((uint128_t)1 << 89) + ((uint128_t)1 << 90) + ((uint128_t)1 << 91);
        mask = mask ^ outside;
    }

    if (num_rows == 9) {
        BBType outside = ((uint128_t)1 << 6) + ((uint128_t)1 << 7) + ((uint128_t)1 << 8) + ((uint128_t)1 << 9) +
                         ((uint128_t)1 << 21) + ((uint128_t)1 << 32) + ((uint128_t)1 << 43) + ((uint128_t)1 << 54) +
                         ((uint128_t)1 << 66) + ((uint128_t)1 << 77) + ((uint128_t)1 << 88) + ((uint128_t)1 << 99) +
                         ((uint128_t)1 << 111) + ((uint128_t)1 << 112) + ((uint128_t)1 << 113) + ((uint128_t)1 << 114);
        mask = mask ^ outside;
    }
    return mask;
}

ChineseCheckers::ChineseCheckers(int num_rows, int num_cols, int num_pieces) {
    this->num_rows_ = num_rows;
    this->num_cols_ = num_cols;
    this->num_pieces_ = num_pieces;
    this->destinations_mask = generate_destinations_mask(num_rows, num_cols);
    this->empties_mask = generate_empties_mask(num_rows, num_cols, num_pieces);
}

void ChineseCheckers::reset(StateType &state) {
    // Create an empty board and add initial configuration of the 4 player
    // pieces.
    // NOTE: This is for 64-bit integers only
    BBType bb_1 = (uint128_t)0;
    BBType bb_2 = (uint128_t)0;
    BBType bit = (uint128_t)1 << (MAX_ROW + 1);
    BBType back_bit = (uint128_t)1 << 108;
    back_bit = back_bit >> ((9 - state.get_num_rows()) * 12);

    // Lookup initial setup for player one.
    // bb_1 = SETUPS[state.get_num_pieces() - 1];
    // bb_1 = generate_initial_configuration(state.get_num_pieces());
    bb_1 = ~this->destinations_mask &
           generate_initial_configuration(state.get_num_pieces());

    // Mirror bits for player two.
    for (int i = 0; i < state.get_num_rows(); i++) {
        for (int j = 0; j < state.get_num_cols(); j++) {
            if (bb_1 & bit) {
                bb_2 = bb_2 | back_bit;
            }
            bit = bit << 1;
            back_bit = back_bit >> 1;
        }
        bit = bit << (MAX_ROW - state.get_num_cols());
        back_bit = back_bit >> (MAX_ROW - state.get_num_cols());
    }
    bb_2 = ~this->destinations_mask & bb_2;
    state.piece_locations =
        get_initial_piece_locs(state.get_num_rows(), state.get_num_pieces());

    initial_board = StateType::BoardType({bb_1, bb_2});
    state.set_board(initial_board);
    state.set_player(Player::One);
}

void ChineseCheckers::print_mask(BBType mask) {
    int shift;
    BBType start = 1;
    BBType bit = (uint128_t)1;

    shift = MAX_ROW;
    for (int i = 0; i < MAX_ROW; i++) {
        for (int j = 0; j < MAX_ROW - i; j++)
            std::cout << " ";
        bit = start;
        for (int j = 0; j <= i; j++) {
            if (mask & bit)
                std::cout << BLUE << FILL_CIRCLE << " " << RESET;
            else
                std::cout << CIRCLE << " ";
            bit = bit >> (MAX_ROW - 1);
        }
        start = start << shift;
        std::cout << "\n";
    }

    start = (uint128_t)1 << 110;
    shift = 1;
    for (int i = 0; i < (MAX_ROW - 1); i++) {
        for (int j = 0; j <= i + 1; j++)
            std::cout << " ";
        bit = start;
        for (int j = (MAX_ROW - 1); j > i; j--) {
            if (mask & bit)
                std::cout << BLUE << FILL_CIRCLE << " " << RESET;
            else
                std::cout << CIRCLE << " ";
            bit = bit >> (MAX_ROW - 1);
        }
        start = start << shift;
        std::cout << "\n";
    }
    std::cout << std::endl;
}

BBType shift(BBType source, int dir) {
    if (dir < 3)
        source = source >> STEP_SHIFTS[dir];
    else
        source = source << STEP_SHIFTS[dir];
    return source;
}

BBType ChineseCheckers::get_steps(const StateType::BoardType board,
                                  int source) const {
    BBType steps = 0;
    BBType source_bit = (uint128_t)1 << source;
    BBType occupied = board[Player::One] | board[Player::Two];

    for (int i = 0; i < 6; i++) {
        steps = steps | shift(source_bit, i);
    }

    steps = steps & ~(this->destinations_mask | occupied);

    return steps;
}

BBType ChineseCheckers::is_hop(StateType::BoardType board, BBType source_bits,
                               int dir) const {
    BBType hop = 0;
    BBType occupied = board[Player::One] | board[Player::Two];

    hop = occupied & shift(source_bits, dir);
    hop = shift(hop, dir) & ~(this->empties_mask | occupied);

    return hop;
}

BBType ChineseCheckers::get_hops(StateType::BoardType board, int source) const {
    BBType hops = 0;
    BBType source_bit = 1;
    source_bit = source_bit << source;
    BBType new_hops = source_bit;
    BBType occupied = board[Player::One] | board[Player::Two];

    while (new_hops != hops) {
        hops = new_hops;
        for (int dir = 0; dir < 6; dir++) {
            new_hops = new_hops | is_hop(board, hops, dir);
        }
    }
    hops = hops & ~(this->destinations_mask | occupied);
    return hops;
}

int location_to_index(int location, int num_rows) {
    // Takes a bit location and returns its corresponding index in an array that
    // would represent the board with dimension num_rows.

    location -= (MAX_ROW + 1);
    int row = (location / MAX_ROW);
    int col = location % MAX_ROW;
    int index = (row * num_rows) + col;
    return index;
}

int index_to_location(int index, int num_rows) {
    int row = index / num_rows;
    int col = index % num_rows;
    int location = (row * MAX_ROW) + col;
    location += (MAX_ROW + 1);
    return location;
}

std::vector<ChineseCheckers::ActionType>
ChineseCheckers::get_actions(const StateType &state) const {
    std::vector<ChineseCheckers::ActionType> actions;
    BBType bit = (uint128_t)0;
    std::vector<int> sources;
    if (state.get_player() == Player::One)
        sources = state.piece_locations[0];
    else
        sources = state.piece_locations[1];

    for (int s : sources) {
        BBType steps, hops;
        steps = get_steps(state.get_board(), s);
        hops = get_hops(state.get_board(), s);
        int base = location_to_index(s, num_rows_);

        // Collate all step actions
        bit = (uint128_t)1;
        for (int i = 0; i < sizeof(BBType) * 8; i++) {
            if (bit & steps) {
                int offset = location_to_index(i, num_rows_);
                ChineseCheckers::ActionType action =
                    (base * num_rows_ * num_rows_) + offset;
                actions.push_back(action);
            }
            bit <<= 1;
        }

        // Collate all step actions
        bit = (uint128_t)1;
        for (int i = 0; i < sizeof(BBType) * 8; i++) {
            if (bit & hops) {
                int offset = location_to_index(i, num_rows_);
                ChineseCheckers::ActionType action =
                    (base * num_rows_ * num_rows_) + offset;
                actions.push_back(action);
            }
            bit <<= 1;
        }
    }

    return actions;
}

std::vector<ChineseCheckers::ActionType>
ChineseCheckers::get_reverse_actions(const StateType &state) const {
    std::vector<ChineseCheckers::ActionType> actions;
    // StateType state_ = state;
    // state_.set_player(state.get_opponent());
    // actions = get_actions(state_);
    BBType bit = (uint128_t)0;
    std::vector<int> sources;
    int inverter = std::pow(state.get_num_cols(), 4) - 1;
    if (state.get_opponent() == Player::One)
        sources = state.piece_locations[0];
    else
        sources = state.piece_locations[1];

    for (int s : sources) {
        BBType steps, hops;
        steps = get_steps(state.get_board(), s);
        hops = get_hops(state.get_board(), s);
        int base = location_to_index(s, num_rows_);

        // Collate all step actions
        bit = (uint128_t)1;
        for (int i = 0; i < sizeof(BBType) * 8; i++) {
            if (bit & steps) {
                int offset = location_to_index(i, num_rows_);
                ChineseCheckers::ActionType action =
                    (offset * num_rows_ * num_rows_) + base;
                actions.push_back(action);
            }
            bit <<= 1;
        }

        // Collate all step actions
        bit = (uint128_t)1;
        for (int i = 0; i < sizeof(BBType) * 8; i++) {
            if (bit & hops) {
                int offset = location_to_index(i, num_rows_);
                ChineseCheckers::ActionType action =
                    (offset * num_rows_ * num_rows_) + base;
                actions.push_back(action);
            }
            bit <<= 1;
        }
    }

    return actions;
}

bool ChineseCheckers::has_actions(const StateType &state) {
    return get_actions(state).size() > 0;
}

// TODO: Add checks for both apply_action and undo_action to ensure actions are
// valid.
int ChineseCheckers::apply_action(StateType &state, ActionType action) {
    // Validation checks:
    // - is the action one for the current player?
    // - does the source piece exist?
    // - is the destination location available?

    int base = action / (num_rows_ * num_rows_);
    int offset = action % (num_rows_ * num_rows_);
    int start = index_to_location(base, num_rows_);
    int end = index_to_location(offset, num_rows_);
    // std::cout << start << "->" << end << std::endl;

    BBType source, destination;
    source = (uint128_t)1 << start;
    destination = (uint128_t)1 << end;

    ChineseCheckersState::BoardType board = state.get_board();
    board[state.get_player()] &= ~source;
    board[state.get_player()] |= destination;

    state.set_board(board);

    return 0;
}

int ChineseCheckers::undo_action(StateType &state, ActionType action) {
    int base = action / (num_rows_ * num_rows_);
    int offset = action % (num_rows_ * num_rows_);
    int end = index_to_location(base, num_rows_);
    int start = index_to_location(offset, num_rows_);
    // std::cout << start << "->" << end << std::endl;

    BBType source, destination;
    source = (uint128_t)1 << start;
    destination = (uint128_t)1 << end;

    ChineseCheckersState::BoardType board = state.get_board();
    board[state.get_opponent()] &= ~source;
    board[state.get_opponent()] |= destination;

    state.set_board(board);

    return 0;
}

ChineseCheckers::StateType
ChineseCheckers::get_next_state(const StateType &state, ActionType action) {
    StateType next_state = state;
    apply_action(next_state, action);
    if (state.get_player() == Player::One)
        next_state.set_player(Player::Two);
    else
        next_state.set_player(Player::One);
    return next_state;
}

ChineseCheckers::StateType
ChineseCheckers::get_previous_state(const StateType &state, ActionType action) {
    StateType previous_state = state;
    undo_action(previous_state, action);
    if (state.get_player() == Player::One)
        previous_state.set_player(Player::Two);
    else
        previous_state.set_player(Player::One);
    return previous_state;
}

bool ChineseCheckers::is_winner(const StateType &state, Player player) const {
    // Checks if the state is a win for the player passed as an argument

    Player opponent = ((player == Player::One) ? Player::Two : Player::One);

    // Is goal filled with opponent pieces?
    if (state.get_board()[opponent] != initial_board[opponent]) {
        BBType joint_board =
            state.get_board()[Player::One] | state.get_board()[Player::Two];
        // Is goal filled and at least one piece belongs to player?
        // std::cout << joint_board << " " << initial_board[opponent] <<
        // std::endl;
        if ((joint_board & initial_board[opponent]) == initial_board[opponent])
            return true;
        else
            return false;
    } else
        return false;
}

bool ChineseCheckers::is_draw(const StateType &state) {
    // Draws occur when states are repeated. A state in isolation cannot be a
    // draw.
    // Check for draws through the use of the solver.
    return false;
}

bool ChineseCheckers::is_terminal(const StateType &state) {
    if (is_winner(state, Player::One))
        return true;
    else if (is_winner(state, Player::Two))
        return true;
    else
        return false;
}

bool ChineseCheckers::is_legal(const StateType &state) {
    // State where both players simultaneously are winning are illegal
    if (is_winner(state, Player::One) && is_winner(state, Player::Two))
        return true;
    return true;
}

ChineseCheckers::Outcomes ChineseCheckers::get_outcome(const StateType &state) {
    if (is_winner(state, Player::One))
        return Outcomes::P1Win;
    if (is_winner(state, Player::Two))
        return Outcomes::P2Win;
    return Outcomes::NonTerminal;
}

// ==============================
// Functions for AlphaZero
// ==============================

std::vector<std::uint8_t>
ChineseCheckers::legal_moves_mask(const StateType &state) {
    std::vector<std::uint8_t> mask(std::pow(state.get_num_rows(), 4));
    std::vector<ActionType> actions = get_actions(state);
    for (auto action : actions)
        mask[action] = 1;
    return mask;
}

std::vector<float> ChineseCheckers::decode_policy(const StateType &state,
                                                  std::vector<float> policy) {
    if (state.get_player() == Player::Two) {
        std::reverse(policy.begin(), policy.end());
        return policy;
    } else
        return policy;
}
