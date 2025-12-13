#include "../include/constants.h"
#include "tic_tac_toe_state.h"
#include <iostream>
#include <cstdint>

TicTacToeState::TicTacToeState() = default;

void TicTacToeState::print_bitboard() const {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int bit = (row * 3) + col;
            if ((board_[0] >> bit) & 1)
                std::cout << GREEN << "x " << RESET;
            else if ((board_[1] >> bit) & 1)
                std::cout << RED << "o " << RESET;
            else
                std::cout << ". ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void TicTacToeState::set_bitboard(std::array<std::uint16_t, 2> board) {
    this->board_ = board;
    // this->board_[1] = board[1];
}

std::string TicTacToeState::state_to_string() { return "";}

void TicTacToeState::string_to_state(std::string state_str) {};
