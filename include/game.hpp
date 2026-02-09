#ifndef GAME_HPP
#define GAME_HPP

#include <vector>
#include <cstdint>

template <typename G>
concept Game = requires(G g, G::StateType state, G::ActionType action,
                        G::StateType::Player player, G::Outcomes outcomes) {
    { g.get_id() } -> std::same_as<std::string>;
    {
        g.get_actions(state)
    } -> std::same_as<std::vector<typename G::ActionType>>;
    { g.apply_action(state, action) } -> std::same_as<int>;
    { g.undo_action(state, action) } -> std::same_as<int>;
    { g.reset(state) } -> std::same_as<void>;
    { g.is_winner(state, player) } -> std::same_as<bool>;
    { g.is_terminal(state) } -> std::same_as<bool>;
    { g.get_outcome(state) } -> std::same_as<typename G::Outcomes>;
    { g.legal_moves_mask(state) } ->std::same_as<std::vector<std::uint8_t>>;
};

#endif
