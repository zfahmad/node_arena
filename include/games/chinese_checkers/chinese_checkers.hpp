#ifndef CHINESE_CHECKERS_HPP
#define CHINESE_CHECKERS_HPP

#include <game.hpp>
#include <games/chinese_checkers/chinese_checkers_state.hpp>
#include <vector>

const int SHIFTS[] = {1, 12, 11, 10, 1, 12, 11, 10};

// const ChineseCheckersState::BBType SHIFT_MASKS[] = {
//     0xFEFEFEFEFEFEFEFE, // Right Shift
//     0xFEFEFEFEFEFEFE00, // Down-right Shift
//     0xFFFFFFFFFFFFFF00, // Down shift
//     0x7F7F7F7F7F7F7F00, // Down-left shift
//     0x7F7F7F7F7F7F7F7F, // Left shift
//     0x007F7F7F7F7F7F7F, // Up-left shift
//     0x00FFFFFFFFFFFFFF, // Up shift
//     0x00FEFEFEFEFEFEFE, // Up-right shift
// };

const int STEP_SHIFTS[] = {
    1, // North-West
    11, // North-East
    10, // East
    1, // South-East
    11, // South-West
    10, // West
};

const int STARTING_LOCATIONS[10][10] = {
    {12, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {13, 23, -1, -1, -1, -1, -1, -1, -1, -1},
    {12, 13, 23, -1, -1, -1, -1, -1, -1, -1},
    {12, 13, 23, 24, -1, -1, -1, -1, -1, -1},
    {12, 13, 23, 34, 14, -1, -1, -1, -1, -1},
    {12, 13, 23, 24, 34, 14, -1, -1, -1, -1},
    {13, 14, 23, 24, 25, 34, 35, -1, -1, -1},
    {12, 13, 14, 23, 24, 25, 34, 35, -1, -1},
    {13, 14, 15, 23, 24, 25, 34, 35, 45, -1},
    {12, 13, 14, 15, 23, 24, 25, 34, 35, 45},
};

const ChineseCheckersState::BBType SETUPS[] = {
    ((uint128_t)1 << 12),

    ((uint128_t)1 << 13) + ((uint128_t)1 << 23),

    ((uint128_t)1 << 12) + ((uint128_t)1 << 13) + ((uint128_t)1 << 23),

    ((uint128_t)1 << 12) + ((uint128_t)1 << 13) + ((uint128_t)1 << 23) +
        ((uint128_t)1 << 24),

    ((uint128_t)1 << 12) + ((uint128_t)1 << 13) + ((uint128_t)1 << 23) +
        ((uint128_t)1 << 34) + ((uint128_t)1 << 14),

    ((uint128_t)1 << 12) + ((uint128_t)1 << 13) + ((uint128_t)1 << 23) +
        ((uint128_t)1 << 24) + ((uint128_t)1 << 34) + ((uint128_t)1 << 14),

    ((uint128_t)1 << 13) + ((uint128_t)1 << 14) + ((uint128_t)1 << 23) +
        ((uint128_t)1 << 24) + ((uint128_t)1 << 25) + ((uint128_t)1 << 34) +
        ((uint128_t)1 << 35),

    ((uint128_t)1 << 12) + ((uint128_t)1 << 13) + ((uint128_t)1 << 14) +
        ((uint128_t)1 << 23) + ((uint128_t)1 << 24) + ((uint128_t)1 << 25) +
        ((uint128_t)1 << 34) + ((uint128_t)1 << 35),

    ((uint128_t)1 << 45) + ((uint128_t)1 << 13) + ((uint128_t)1 << 14) +
        ((uint128_t)1 << 23) + ((uint128_t)1 << 24) + ((uint128_t)1 << 25) +
        ((uint128_t)1 << 34) + ((uint128_t)1 << 35) + ((uint128_t)1 << 15),

    ((uint128_t)1 << 12) + ((uint128_t)1 << 45) + ((uint128_t)1 << 13) +
        ((uint128_t)1 << 14) + ((uint128_t)1 << 23) + ((uint128_t)1 << 24) +
        ((uint128_t)1 << 25) + ((uint128_t)1 << 34) + ((uint128_t)1 << 35) +
        ((uint128_t)1 << 15),
};

class ChineseCheckers {
public:
    enum class Outcomes : std::int8_t { NonTerminal, P1Win, P2Win, Draw };
    using ActionType = int;
    using StateType = ChineseCheckersState;

    ChineseCheckers(int num_rows = 6, int num_cols = 6, int num_pieces = 6);
    std::string get_id() { return "chinese_checkers"; }
    std::vector<ActionType> get_actions(const StateType &state) const;
    std::vector<ActionType> get_reverse_actions(const StateType &state) const;
    bool has_actions(const StateType &state);
    int apply_action(StateType &state, ActionType action);
    int undo_action(StateType &state, ActionType action);
    StateType get_next_state(const StateType &state, ActionType action);
    StateType get_previous_state(const StateType &state, ActionType action);
    void reset(StateType &state);
    bool is_winner(const StateType &state, StateType::Player player) const;
    bool is_draw(const StateType &state);
    bool is_terminal(const StateType &state);
    bool is_legal(const StateType &state);
    Outcomes get_outcome(const StateType &state);
    std::vector<std::uint8_t> legal_moves_mask(const StateType &state);
    std::vector<float> decode_policy(const StateType &state,
                                     std::vector<float> policy);
    void print_mask(StateType::BBType mask);
    StateType::BBType get_steps(StateType::BoardType board, int source) const;
    StateType::BBType is_hop(StateType::BoardType board,
                             ChineseCheckersState::BBType source_bits,
                             int dir) const;
    StateType::BBType get_hops(StateType::BoardType board, int source) const;
    StateType::BBType destinations_mask;
    StateType::BBType empties_mask;
    StateType::BoardType initial_board;

protected:
private:
    int num_rows_, num_cols_, num_pieces_;
};

static_assert(Game<ChineseCheckers>);

#endif
