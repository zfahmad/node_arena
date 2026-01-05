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
    std::vector<int> get_actions(const ConnectFourState &state) const;
    int apply_action(ConnectFourState &state, int action);
    int undo_action(ConnectFourState &state, int action);
    void reset(ConnectFourState &state);
    bool is_winner(const ConnectFourState &state, StateType::Player player);
    bool is_draw(const ConnectFourState &state);
    ConnectFourState get_next_state(const ConnectFourState &state, int action);
};

static_assert(Game<ConnectFour>);

#endif
