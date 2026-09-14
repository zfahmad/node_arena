#ifndef LINES_OF_ACTION_STATE_HPP
#define LINES_OF_ACTION_STATE_HPP

#include <cstdint>
#include <player.hpp>
#include <state.hpp>
#include <vector>

#define ROW_MAX 8

class LinesOfActionState {
public:
    enum class Player { One, Two };
    using BBType = std::uint64_t;
    using BoardType = PlayerIndexed<BBType, Player>;

    LinesOfActionState(int num_rows = 8) : num_rows_ ( num_rows ) {};
    void print_board();
    const BoardType &get_board() const { return {board_}; }
    void set_board(BoardType board);
    std::vector<BBType> to_compact() const;
    void from_compact(std::vector<BBType>);
    std::vector<std::vector<std::uint8_t>> to_array();
    std::string to_string();
    void from_string(const std::string state_str);
    Player get_player() const { return player_; };
    Player get_opponent() const {
        return (player_ == Player::One) ? Player::Two : Player::One;
    }
    void set_player(Player player) { player_ = player; }
    int get_num_rows() const { return num_rows_; }
    int num_pieces(BBType board) const;

    std::array<std::uint64_t, 2> canonical_form();
    void from_canonical_form(std::array<BBType, 2> canonical_state);

    bool operator==(const LinesOfActionState &) const = default;

protected:
    BoardType board_ = BoardType();
    Player player_ = Player::One;
    int num_rows_;
};

static_assert(State<LinesOfActionState>);

#endif
