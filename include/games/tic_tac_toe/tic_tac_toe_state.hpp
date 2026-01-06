#ifndef TIC_TAC_TOE_STATE_HPP
#define TIC_TAC_TOE_STATE_HPP

#include <player.hpp>
#include <cstdint>
#include <state.hpp>

class TicTacToeState {
public:
    enum class Player { One, Two };
    using BoardType = PlayerIndexed<std::uint16_t, Player>;

    void print_board();
    TicTacToeState();
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

static_assert(State<TicTacToeState>);

#endif
