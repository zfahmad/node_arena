#ifndef OTHELLO_STATE_HPP
#define OTHELLO_STATE_HPP

#include <player.hpp>
#include <cstdint>
#include <state.hpp>

class OthelloState {
public:
    enum class Player { One, Two };
    using BBType = std::uint64_t;
    using BoardType = PlayerIndexed<BBType, Player>;

    OthelloState(int num_rows = 8, int num_cols = 8);
    void print_board();
    const BoardType &get_board() const { return {board_}; }
    void set_board(BoardType board);
    std::string state_to_string();
    void string_to_state(const std::string state_str);
    Player get_player() const { return player_; };
    Player get_opponent() const;
    void set_player(Player player) { player_ = player; }
    int get_num_cols() const { return num_cols_; }
    int get_num_rows() const { return num_rows_; }

protected:
    BoardType board_ = BoardType();
    Player player_ = Player::One;
    int num_rows_, num_cols_;
};

static_assert(State<OthelloState>);

#endif
