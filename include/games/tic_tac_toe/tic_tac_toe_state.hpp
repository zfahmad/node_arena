#ifndef TIC_TAC_TOE_STATE_HPP
#define TIC_TAC_TOE_STATE_HPP

#include <player.hpp>
#include <state.hpp>

class TicTacToeState {
public:
    enum class Player { One, Two };
    using BBType = std::uint16_t;
    using BoardType = PlayerIndexed<BBType, Player>;

    TicTacToeState();
    void print_board();
    const BoardType &get_board() const { return {board_}; }
    void set_board(BoardType board);
    std::string state_to_string();
    void string_to_state(const std::string state_str);
    std::vector<std::vector<std::uint8_t>> to_array();
    Player get_player() const { return player_; };
    Player get_opponent() const {
        return (player_ == Player::One) ? Player::Two : Player::One;
    }
    void set_player(Player player) { player_ = player; }

protected:
    BoardType board_ = BoardType();
    Player player_ = Player::One;
};

static_assert(State<TicTacToeState>);

#endif
