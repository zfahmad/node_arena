#ifndef TIC_TAC_TOE_STATE_HPP
#define TIC_TAC_TOE_STATE_HPP

#include <player.hpp>
#include <state.hpp>

class TicTacToeState {
public:
    enum class Player { One, Two };
    using BBType = std::uint16_t;
    using BoardType = PlayerIndexed<BBType, Player>;

    TicTacToeState(int num_rows=3, int num_cols=3);
    void print_board();
    const BoardType &get_board() const { return {board_}; }
    void set_board(BoardType board);
    std::vector<BBType> to_compact() const;
    void from_compact(std::vector<BBType>);
    std::string to_string();
    void from_string(const std::string state_str);
    std::vector<std::vector<std::uint8_t>> to_array();
    Player get_player() const { return player_; };
    Player get_opponent() const {
        return (player_ == Player::One) ? Player::Two : Player::One;
    }
    void set_player(Player player) { player_ = player; }

protected:
    BoardType board_ = BoardType();
    Player player_ = Player::One;
    int num_rows_, num_cols_ = 3;
};

static_assert(State<TicTacToeState>);

#endif
