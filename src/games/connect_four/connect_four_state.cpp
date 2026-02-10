#include <charconv>
#include <constants.hpp>
#include <games/connect_four/connect_four_state.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

ConnectFourState::ConnectFourState(int num_rows, int num_cols) {
    // WARN: Min size shouldn't be greater than 0. This is only for testing.
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
    std::cout << " ";
    for (int i = 0; i < this->num_cols_; i++) {
        std::cout << GRAY << i << " ";
    }
    std::cout << RESET << "\n";
    for (int row = 0; row < this->num_rows_; row++) {
        std::cout << " ";
        for (int col = 0; col < this->num_cols_; col++) {
            if (board_[Player::One] & bit)
                // std::cout << GREEN << "x " << RESET;
                // std::cout << GREEN << CROSS << " " << RESET;
                std::cout << BLUE << FILL_CIRCLE << " " << RESET;
            else if (board_[Player::Two] & bit)
                // std::cout << RED << "o " << RESET;
                // std::cout << RED << NAUGHT << " " << RESET;
                std::cout << RED << FILL_CIRCLE << " " << RESET;
            else
                // std::cout << GRAY << ". " << RESET;
                // std::cout << GRAY << DOT << " " << RESET;
                std::cout << CIRCLE << " ";
            bit = (bit << 1);
        }
        std::cout << "\n";
        bit = (bit << 1);
    }
    std::cout << std::endl;
}

void ConnectFourState::set_board(BoardType board) { this->board_ = board; }

std::vector<ConnectFourState::BBType> ConnectFourState::to_compact() const {
    std::vector<BBType> board;
    board.reserve(2);
    board.push_back(board_[Player::One]);
    board.push_back(board_[Player::Two]);
    return board;
}

void ConnectFourState::from_compact(std::vector<BBType> compact_board) {
    if ((compact_board[0] & compact_board[1]) != 0)
        throw std::logic_error("Bit collision");
    board_[Player::One] = compact_board[0];
    board_[Player::Two] = compact_board[1];
}

std::vector<std::vector<std::uint8_t>> ConnectFourState::to_array() {
    std::vector<std::vector<uint8_t>> arrs;
    arrs.reserve(2);
    std::vector<Player> players;
    players.push_back(get_player());
    players.push_back(get_opponent());

    for (Player p : players) {
        std::vector<uint8_t> player_arr;
        player_arr.reserve(num_cols_ * num_rows_);
        BBType bits = board_[p];
        bits = bits >> (num_cols_ + 1);
        for (int i = 0; i < num_rows_; i++) {
            for (int j = 0; j < num_cols_; j++)
                player_arr.push_back((bits >> ((i * (num_cols_ + 1)) + j)) & 1);
        }
        arrs.push_back(std::move(player_arr));
    }

    return arrs;
}

std::string ConnectFourState::to_string() {
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

void ConnectFourState::from_string(std::string state_str) {
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
