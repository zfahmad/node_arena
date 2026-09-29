#ifndef LINES_OF_ACTION_HPP
#define LINES_OF_ACTION_HPP

#include <game.hpp>
#include <games/lines_of_action/lines_of_action_state.hpp>
#include <vector>

enum dirs {
    NORTH,
    NORTHEAST,
    EAST,
    SOUTHEAST,
    SOUTH,
    SOUTHWEST,
    WEST,
    NORTHWEST
};

const int SHIFTS[8] = {
    -8,
    -7,
    1,
    9,
    8,
    7,
    -1,
    -9
};

const LinesOfActionState::BBType SHIFT_MASKS[] = {
    0x00FFFFFFFFFFFFFF, // Up shift
    0x00FEFEFEFEFEFEFE, // Up-right shift
    0xFEFEFEFEFEFEFEFE, // Right Shift
    0xFEFEFEFEFEFEFE00, // Down-right Shift
    0xFFFFFFFFFFFFFF00, // Down shift
    0x7F7F7F7F7F7F7F00, // Down-left shift
    0x7F7F7F7F7F7F7F7F, // Left shift
    0x007F7F7F7F7F7F7F, // Up-left shift
};

class LinesOfAction {
public:
    enum class Outcomes : std::int8_t { NonTerminal, P1Win, P2Win, Draw };
    using ActionType = int;
    using StateType = LinesOfActionState;

    LinesOfAction(int num_rows);
    std::string get_id() { return "lines_of_action"; }
    std::vector<ActionType> get_actions(const StateType &state) const;
    std::vector<ActionType> get_reverse_actions(const StateType &state) const;
    int apply_action(StateType &state, ActionType action);
    int undo_action(StateType &state, ActionType action);
    StateType get_next_state(const StateType &state, ActionType action);
    StateType get_previous_state(const StateType &state, ActionType action);
    void reset(StateType &state);
    bool is_winner(const StateType &state, StateType::Player player);
    bool is_draw(const StateType &state);
    bool is_terminal(const StateType &state);
    Outcomes get_outcome(const StateType &state);
    std::vector<std::uint8_t> legal_moves_mask(const StateType &state);
    std::vector<float> decode_policy(const StateType &state,
                                     std::vector<float> policy);

    // Lines of Action specific functions
    bool shift_check(StateType::BBType board, int direction);
    StateType::BBType dest_table[ROW_MAX][ROW_MAX * ROW_MAX][ROW_MAX];
    StateType::BBType path_table[ROW_MAX][ROW_MAX * ROW_MAX][ROW_MAX];
    StateType::BBType valid;

private:
    int num_rows_;
};

std::array<int, 3> action_to_inds(int index, int num_rows);
int index_to_location(int index, int num_rows);

static_assert(Game<LinesOfAction>);

#endif
