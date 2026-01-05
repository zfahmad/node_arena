#include <constants.hpp>
#include <games/tic_tac_toe/tic_tac_toe_state.hpp>
#include <iostream>

TicTacToeState::TicTacToeState() = default;

void TicTacToeState::print_board() {
    // Print the board to screen.
    // LSB of the bit representation is top-left cell;
    // MSB of the bit representation is bottom-right.
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int bit = (row * 3) + col;
            if ((board_[Player::One] >> bit) & 1)
                std::cout << GREEN << "x " << RESET;
            else if ((board_[Player::Two] >> bit) & 1)
                std::cout << RED << "o " << RESET;
            else
                std::cout << ". ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void TicTacToeState::set_board(BoardType board) { this->board_ = board; }

std::string TicTacToeState::state_to_string() {
    // Converts the state representation to readable string.
    // First nine characters represent the board.
    // Last character is the current player at the state.
    std::string state_str = "";
    for (int i = 0; i < 9; i++) {
        if (board_[Player::One] & 1)
            state_str += '1';
        else if (board_[Player::Two] & 1)
            state_str += '2';
        else
            state_str += '0';
        board_[Player::One] = (board_[Player::One] >> 1);
        board_[Player::Two] = (board_[Player::Two] >> 1);
    }
    state_str += static_cast<char>(player_);
    return state_str;
}

void TicTacToeState::string_to_state(std::string state_str) {
    board_[Player::One] = 0;
    board_[Player::Two] = 0;
    int count = 0;
    for (int i = 0; i < 9; i++) {
        if (state_str[i] == '1')
            (board_[Player::One] |= (1L << count));
        else if (state_str[i] == '2')
            (board_[Player::Two] |= (1L << count));
        count++;
    }
    if (state_str.back() == '0')
        player_ = Player::One;
    else
        player_ = Player::Two;
};
