#include <cassert>
#include <charconv>
#include <constants.hpp>
#include <games/othello/othello_state.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

OthelloState::OthelloState(int num_rows, int num_cols) {
    if ((num_rows < 4) || (num_rows > 8) || (num_cols < 4) || (num_cols > 8)) {
        throw std::invalid_argument(
            "Board dimensions must be valid sizes between 4 and 8.");
    }
    if ((num_rows % 2) | (num_cols % 2)) {
        throw std::invalid_argument("Board dimensions must be even.");
    }

    this->num_rows_ = num_rows;
    this->num_cols_ = num_cols;
    this->board_ = {};
}

void OthelloState::print_board() {
    // Print the board to screen.
    // Skips the first num_cols bits as they are intended to be 0
    // LSB of the bit representation is top-left cell;
    // MSB of the bit representation is bottom-right.
    BBType bit = 1ULL;
    for (int row = 0; row < this->num_rows_; row++) {
        std::cout << GRAY << std::setw(3) << (row * 8) << " " << RESET;
        for (int col = 0; col < this->num_cols_; col++) {
            if (board_[Player::One] & bit)
                std::cout << BLUE << FILL_CIRCLE << " " << RESET;
            else if (board_[Player::Two] & bit)
                std::cout << RED << FILL_CIRCLE << " " << RESET;
            else
                std::cout << CIRCLE << " ";
            bit = (bit << 1);
        }
        std::cout << "\n";
        bit = (bit << (8 - this->get_num_cols()));
    }
    std::cout << std::endl;
}

void OthelloState::set_board(BoardType board) { this->board_ = board; }

std::string OthelloState::state_to_string() {
    // Converts the state representation to a string.
    // First sixteen characters represent the board for player one in hex.
    // First sixteen characters represent the board for player two in hex.
    // Last character is the current player at the state.
    std::string state_str = "";

    std::stringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(2 * sizeof(BBType))
           << board_[Player::One];
    stream << std::hex << std::setfill('0') << std::setw(2 * sizeof(BBType))
           << board_[Player::Two];
    state_str += stream.str();

    if (player_ == Player::One)
        state_str += "0";
    else
        state_str += "1";
    state_str +=
        std::to_string(this->num_rows_) + std::to_string(this->num_cols_);

    return state_str;
}

void OthelloState::string_to_state(std::string state_str) {
    const char *data = state_str.data();
    auto r1 = std::from_chars(data, data + 16, board_[Player::One], 16);
    auto r2 = std::from_chars(data + 16, data + 32, board_[Player::Two], 16);

    assert((!(board_[Player::One] & board_[Player::Two])) &&
           "State has overlapping pieces.");

    if (state_str[32] == '0') {
        assert(!(num_pieces(board_[Player::One]) +
                 num_pieces(board_[Player::Two]) % 2) &&
               "Number of pieces must be even for player one to be acting.");
        player_ = Player::One;
    } else {
        assert((num_pieces(board_[Player::One]) +
                num_pieces(board_[Player::Two]) % 2) &&
               "Number of pieces must be odd for player two to be acting.");
        player_ = Player::Two;
    }

    this->num_rows_ = state_str[33] - '0';
    this->num_cols_ = state_str[34] - '0';
    assert((this->num_rows_ >= 4) & (this->num_rows_ <= 8) &
               !(this->num_rows_ % 2) &&
           "Invalid number of rows: Must be 4, 6, or 8.");
    assert((this->num_cols_ >= 4) & (this->num_cols_ <= 8) &
               !(this->num_cols_ % 2) &&
           "Invalid number of rows: Must be 4, 6, or 8.");
};

int OthelloState::num_pieces(BBType board) const {
    // Counts the number of pieces given a specific player's board.
    int count = 0;
    while (board) {
        board &= board - 1;
        count++;
    }
    return count;
}
