#ifndef CONNECT_FOUR_HPP
#define CONNECT_FOUR_HPP

#include <game.hpp>
#include <games/connect_four/connect_four_state.hpp>
#include <vector>

class ConnectFour {
public:
    using ActionType = int;
    using StateType = ConnectFourState;

    ConnectFour();
    std::vector<ActionType> get_actions(const ConnectFourState &state) const;
    int apply_action(ConnectFourState &state, ActionType action);
    int undo_action(ConnectFourState &state, ActionType action);
    void reset(ConnectFourState &state);
    bool is_winner(const ConnectFourState &state, StateType::Player player);
    bool is_draw(const ConnectFourState &state);
    ConnectFourState get_next_state(const ConnectFourState &state,
                                    ActionType action);

    // Connect Four specific functions
    bool shift_check(ConnectFourState::BBType board, int direction);
};

static_assert(Game<ConnectFour>);

#endif
