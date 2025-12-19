#ifndef TIC_TAC_TOE_STATE_H
#define TIC_TAC_TOE_STATE_H

#include "../include/state.h"
#include <array>
#include <cstdint>

class TicTacToeState {
public:
    using BoardType = std::array<std::uint16_t, 2>;
    enum class Player { One, Two };
    void print_board();
    TicTacToeState();
    const BoardType &get_board() const { return {board_}; }
    void set_board(BoardType board);
    std::string state_to_string();
    void string_to_state(const std::string state_str);
    Player get_player() const { return player_; };
    void set_player(Player player) { player_ = player; }

protected:
    BoardType board_ = {0, 0};
    Player player_ = Player::One;
};

static_assert(State<TicTacToeState, std::array<std::uint16_t, 2>>);

#endif
