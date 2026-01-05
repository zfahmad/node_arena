#ifndef CONNECT_FOUR_STATE_HPP
#define CONNECT_FOUR_STATE_HPP

#include "player.hpp"
#include <cstdint>
#include <state.hpp>

class ConnectFourState {
public:
    enum class Player { One, Two };
    using BoardType = PlayerIndexed<std::uint64_t, Player>;

    void print_board();
    ConnectFourState();
    const BoardType &get_board() const { return {board_}; }
    void set_board(BoardType board);
    std::string state_to_string();
    void string_to_state(const std::string state_str);
    Player get_player() const { return player_; };
    void set_player(Player player) { player_ = player; }

protected:
    BoardType board_ = BoardType();
    Player player_ = Player::One;
};

static_assert(State<ConnectFourState>);

#endif
