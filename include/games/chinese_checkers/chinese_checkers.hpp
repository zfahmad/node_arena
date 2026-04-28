#ifndef CHINESE_CHECKERS_HPP
#define CHINESE_CHECKERS_HPP

#include <game.hpp>
#include <games/chinese_checkers/chinese_checkers_state.hpp>
#include <vector>

const int SHIFTS[] = {1, 9, 8, 7, 1, 9, 8, 7};

const ChineseCheckersState::BBType SHIFT_MASKS[] = {
    0xFEFEFEFEFEFEFEFE, // Right Shift
    0xFEFEFEFEFEFEFE00, // Down-right Shift
    0xFFFFFFFFFFFFFF00, // Down shift
    0x7F7F7F7F7F7F7F00, // Down-left shift
    0x7F7F7F7F7F7F7F7F, // Left shift
    0x007F7F7F7F7F7F7F, // Up-left shift
    0x00FFFFFFFFFFFFFF, // Up shift
    0x00FEFEFEFEFEFEFE, // Up-right shift
};

const int STEP_SHIFTS[] = {
    1, // North-West
    8, // North-East
    7, // East
    1, // South-East
    8, // South-West
    7, // West
};

const int STARTING_LOCATIONS[10][10] = {
    {9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 17, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 17, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 17, 18, -1, -1, -1, -1, -1, -1},
    {9, 10, 17, 11, 25, -1, -1, -1, -1, -1},
    {9, 10, 17, 11, 25, 18, -1, -1, -1, -1},
    {10, 17, 11, 25, 18, 19, 26, -1, -1, -1},
    {9, 10, 17, 11, 25, 18, 19, 26, -1, -1},
    {10, 17, 11, 25, 18, 19, 26, 12, 33, -1},
    {9, 10, 17, 11, 25, 18, 19, 26, 12, 33},
};

const ChineseCheckersState::BBType SETUPS[] = {
    (1 << 9),
    (1 << 17) + (1 << 10),
    (1 << 9) + (1 << 17) + (1 << 10),
    (1 << 9) + (1 << 17) + (1 << 10) + (1 << 18),
    (1 << 9) + (1 << 17) + (1 << 10) + (1 << 25) + (1 << 11),
    (1 << 9) + (1 << 17) + (1 << 10) + (1 << 25) + (1 << 18) + (1 << 11),
    (1 << 17) + (1 << 10) + (1 << 25) + (1 << 18) + (1 << 11) + (1 << 26) +
        (1 << 19),
    (1ULL << 9) + (1ULL << 17) + (1ULL << 10) + (1ULL << 25) + (1ULL << 18) +
        (1ULL << 11) + (1ULL << 26) + (1ULL << 19),
    (1ULL << 17) + (1ULL << 10) + (1ULL << 25) + (1ULL << 18) + (1ULL << 11) +
        (1ULL << 33) + (1ULL << 26) + (1ULL << 19) + (1ULL << 12),
    (1ULL << 9) + (1ULL << 17) + (1ULL << 10) + (1ULL << 25) + (1ULL << 18) +
        (1ULL << 11) + (1ULL << 33) + (1ULL << 26) + (1ULL << 19) +
        (1ULL << 12),
};

class ChineseCheckers {
public:
    enum class Outcomes { NonTerminal, P1Win, P2Win, Draw };
    using ActionType = std::array<int, 2>;
    using StateType = ChineseCheckersState;

    ChineseCheckers(int num_rows, int num_cols, int num_pieces);
    std::string get_id() { return "chinese_checkers"; }
    std::vector<ActionType> get_actions(const StateType &state) const;
    bool has_actions(const StateType &state);
    int apply_action(StateType &state, ActionType action);
    int undo_action(StateType &state, ActionType action);
    ChineseCheckersState get_next_state(const StateType &state,
                                        ActionType action);
    void reset(StateType &state);
    bool is_winner(const StateType &state, StateType::Player player);
    bool is_draw(const StateType &state);
    bool is_terminal(const StateType &state);
    Outcomes get_outcome(const StateType &state);
    std::vector<std::uint8_t> legal_moves_mask(const StateType &state);
    void print_mask(StateType::BBType mask);
    StateType::BBType get_steps(StateType::BoardType board, int source) const;
    StateType::BBType is_hop(StateType::BoardType board,
                             ChineseCheckersState::BBType source_bits, int dir) const;
    StateType::BBType get_hops(StateType::BoardType board, int source) const;
    StateType::BoardType initial_board;
    StateType::BBType destinations_mask;
    StateType::BBType empties_mask;

private:
    int num_rows_, num_cols_, num_pieces_;
};

static_assert(Game<ChineseCheckers>);

#endif
