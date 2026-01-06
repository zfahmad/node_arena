#include <charconv>
#include <constants.hpp>
#include <cstdint>
#include <games/connect_four/connect_four_state.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

ConnectFourState::ConnectFourState(int num_rows, int num_cols) {
    if ((num_rows < 0) || (num_rows > 6) || (num_cols < 0) || (num_cols > 7)) {
        throw std::invalid_argument("Board dimensions must be valid sizes.");
    }
    this->num_rows_ = num_rows;
    this->num_cols_ = num_cols;
}

void ConnectFourState::print_board() {
    // Print the board to screen.
    // Skips the first num_cols bits as they are intended to be 0
    // LSB of the bit representation is top-left cell;
    // MSB of the bit representation is bottom-right.
    BBType bit = (1 << (this->num_cols_ + 1));
    for (int row = 0; row < this->num_rows_; row++) {
        for (int col = 0; col < this->num_cols_; col++) {
            if (board_[Player::One] & bit)
                std::cout << GREEN << "x " << RESET;
            else if (board_[Player::Two] & bit)
                std::cout << RED << "o " << RESET;
            else
                std::cout << ". ";
            bit = (bit << 1);
        }
        std::cout << "\n";
        bit = (bit << 1);
    }
    std::cout << std::endl;
}

void ConnectFourState::set_board(BoardType board) { this->board_ = board; }

std::string ConnectFourState::state_to_string() {
    // Converts the state representation to a string.
    // First sixteen characters represent the board for player one in hex.
    // First sixteen characters represent the board for player two in hex.
    // Last character is the current player at the state.
    std::string state_str = "";

    std::stringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(2 * sizeof(uint64_t))
           << board_[Player::One];
    stream << std::hex << std::setfill('0') << std::setw(2 * sizeof(uint64_t))
           << board_[Player::Two];
    state_str += stream.str();

    if (player_ == Player::One)
        state_str += "0";
    else
        state_str += "1";
    state_str += std::to_string(this->num_rows_) + std::to_string(this->num_cols_);

    return state_str;
}

void ConnectFourState::string_to_state(std::string state_str) {
    const char *data = state_str.data();
    auto r1 = std::from_chars(data, data + 16, board_[Player::One], 16);
    auto r2 = std::from_chars(data + 16, data + 32, board_[Player::Two], 16);

    if (state_str[32] == '0')
        player_ = Player::One;
    else
        player_ = Player::Two;
    this->num_rows_ = state_str[33] - '0';
    this->num_cols_ = state_str[34] - '0';
};
