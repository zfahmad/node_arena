#ifndef TIC_TAC_TOE_STATE_H
#define TIC_TAC_TOE_STATE_H

#include "../include/state.h"
#include <array>
#include <cstdint>

class TicTacToeState : public State<TicTacToeState, std::array<std::uint16_t, 2>> {
  public:
    void print_bitboard() const override;
    TicTacToeState();
    const std::array<std::uint16_t, 2> &get_bitboard() const {
        return { board_ };
    }
    void set_bitboard(std::array<std::uint16_t, 2> board) override;
    std::string state_to_string() override;
    void string_to_state(std::string state_str) override;
    int get_player() const override { return player; };

  private:
    std::array<std::uint16_t, 2> board_ = {0, 0};
    int player = -1;
};

#endif
