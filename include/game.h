#ifndef GAME_H
#define GAME_H

#include <vector>
// #include "../include/state.h"

// template <typename Derived, typename StateT, typename ActionT> class Game {
//   public:
//     using StateType = StateT;
//     const std::vector<ActionT> &get_actions(const StateT &state) const {
//         return static_cast<const Derived *>(this)->get_actions();
//     }
//     virtual int apply_action(StateT &state, ActionT action) = 0;
//     virtual int undo_action(StateT &state, ActionT action) = 0;
//     virtual void reset(StateT &state) = 0;
// };

template <typename G, typename StateType, typename ActionType>
concept Game = requires(G g, StateType state, ActionType action, int player) {
    { g.get_actions(state) } -> std::same_as<std::vector<ActionType>>;
    { g.apply_action(state, action) } -> std::same_as<int>;
    { g.undo_action(state, action) } -> std::same_as<int>;
    { g.reset(state) } -> std::same_as<void>;
    { g.is_winner(state, player) } -> std::same_as<bool>;
};

#endif
