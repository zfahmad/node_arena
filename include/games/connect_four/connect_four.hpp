#ifndef CONNECT_FOUR_HPP
#define CONNECT_FOUR_HPP

#include <game.hpp>
#include <games/connect_four/connect_four_state.hpp>
#include <vector>

class ConnectFour {
public:
    enum class Outcomes { NonTerminal, P1Win, P2Win, Draw };
    using ActionType = int;
    using StateType = ConnectFourState;

    ConnectFour();
    std::string get_id() { return "ConnectFour"; }
    std::vector<ActionType> get_actions(const StateType &state) const;
    int apply_action(StateType &state, ActionType action);
    int undo_action(StateType &state, ActionType action);
    StateType get_next_state(const StateType &state, ActionType action);
    void reset(StateType &state);
    bool is_winner(const StateType &state, StateType::Player player);
    bool is_draw(const StateType &state);
    bool is_terminal(const StateType &state);
    Outcomes get_outcome(const StateType &state);
    std::vector<std::uint8_t> legal_moves_mask(const StateType &state);

    // Connect Four specific functions
    bool shift_check(StateType::BBType board, int direction);
};

static_assert(Game<ConnectFour>);

#endif
