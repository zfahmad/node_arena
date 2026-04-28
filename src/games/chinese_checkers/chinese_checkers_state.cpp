#include <cassert>
#include <charconv>
#include <cmath>
#include <constants.hpp>
#include <games/chinese_checkers/chinese_checkers_state.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

ChineseCheckersState::ChineseCheckersState(int num_rows, int num_cols,
                                           int num_pieces) {
    if ((num_rows < 3) || (num_rows > 6) || (num_cols < 3) || (num_cols > 6)) {
        throw std::invalid_argument(
            "Board dimensions must be valid sizes between 3 and 7.");
    }

    this->num_rows_ = num_rows;
    this->num_cols_ = num_cols;
    this->num_pieces_ = num_pieces;
    this->board_ = {};
}

//                0
//               3 1
//              6 4 2
//               7 5
//                8

void ChineseCheckersState::print_board() {
    int shift = 1;
    BBType start = 2;
    BBType bit;

    shift = 8;
    for (int i = 0; i < num_rows_; i++) {
        for (int j = 0; j < num_rows_ - i; j++)
            std::cout << " ";
        start = start << shift;
        bit = start;
        for (int j = 0; j <= i; j++) {
            if (board_[Player::One] & bit)
                std::cout << BLUE << FILL_CIRCLE << " " << RESET;
            else if (board_[Player::Two] & bit)
                std::cout << RED << FILL_CIRCLE << " " << RESET;
            else
                std::cout << CIRCLE << " ";
            bit = bit >> 7;
        }
        std::cout << "\n";
    }

    shift = 1;
    for (int i = 0; i < num_rows_ - 1; i++) {
        for (int j = 0; j <= i + 1; j++)
            std::cout << " ";
        start = start << shift;
        bit = start;
        for (int j = num_cols_ - 1; j > i; j--) {
            if (board_[Player::One] & bit)
                std::cout << BLUE << FILL_CIRCLE << " " << RESET;
            else if (board_[Player::Two] & bit)
                std::cout << RED << FILL_CIRCLE << " " << RESET;
            else
                std::cout << CIRCLE << " ";
            bit = bit >> 7;
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int base_height(int num_pieces) {
    int height = (int)std::ceil((std::sqrt(1 + 8.0 * num_pieces) - 1) / 2);
    return height;
}

std::vector<std::vector<uint8_t>> ChineseCheckersState::to_array() {
    std::vector<std::vector<uint8_t>> arrs;
    arrs.reserve(2);
    std::vector<Player> players;
    players.push_back(get_player());
    players.push_back(get_opponent());

    for (Player p : players) {
        std::vector<uint8_t> player_arr;
        player_arr.reserve(num_cols_ * num_rows_);
        BBType bits = board_[p];
        bits = bits >> 9;
        for (int i = 0; i < num_rows_; i++) {
            for (int j = 0; j < num_cols_; j++) {
                player_arr.push_back(bits & 1);
                bits = bits >> 1;
            }
            bits = bits >> (8 - num_rows_);
        }
        arrs.push_back(std::move(player_arr));
    }
    return arrs;
}

std::vector<int> get_player_locations(ChineseCheckersState::BBType board) {
    std::vector<int> locations;
    ChineseCheckersState::BBType bit = 1ULL;

    for (int i = 0; i < (sizeof(ChineseCheckersState::BBType) * 8); i++) {
        if (bit & board)
            locations.push_back(i);
        bit = bit << 1;
    }

    return locations;
}

void ChineseCheckersState::set_board(BoardType board) {
    this->board_ = board;
    std::vector<std::vector<int>> locations;
    locations.push_back(get_player_locations(board_[Player::One]));
    locations.push_back(get_player_locations(board_[Player::Two]));
    this->piece_locations = std::move(locations);
}

// std::vector<ChineseCheckersState::BBType> ChineseCheckersState::to_compact()
// const {
//     std::vector<BBType> board;
//     board.reserve(2);
//     board.push_back(board_[Player::One]);
//     board.push_back(board_[Player::Two]);
//     return board;
// }
//
// void ChineseCheckersState::from_compact(std::vector<BBType> compact_board) {
//     if ((compact_board[0] & compact_board[1]) != 0)
//         throw std::logic_error("Bit collision");
//     board_[Player::One] = compact_board[0];
//     board_[Player::Two] = compact_board[1];
// }
//
std::string ChineseCheckersState::to_string() {
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

void ChineseCheckersState::from_string(std::string state_str) {
    const char *data = state_str.data();
    auto r1 = std::from_chars(data, data + 16, board_[Player::One], 16);
    auto r2 = std::from_chars(data + 16, data + 32, board_[Player::Two], 16);

    assert((!(board_[Player::One] & board_[Player::Two])) &&
           "State has overlapping pieces.");
    int p1_num_pieces = num_pieces(board_[Player::One]);
    assert(p1_num_pieces == num_pieces(board_[Player::Two]));
    this->num_rows_ = state_str[33] - '0';
    this->num_cols_ = state_str[34] - '0';
    assert((this->num_rows_ >= 4) & (this->num_rows_ <= 7) &
               !(this->num_rows_ % 2) &&
           "Invalid number of rows: Must be between 4 and 7.");
    assert((this->num_cols_ >= 4) & (this->num_cols_ <= 7) &
               !(this->num_cols_ % 2) &&
           "Invalid number of columns: Must be between 4 and 7.");
    this->num_pieces_ = p1_num_pieces;

    std::vector<std::vector<int>> locations;
    locations.push_back(get_player_locations(board_[Player::One]));
    locations.push_back(get_player_locations(board_[Player::Two]));
    this->piece_locations = locations;
}

int ChineseCheckersState::num_pieces(BBType board) const {
    // Counts the number of pieces given a specific player's board.
    int count = 0;
    while (board) {
        board &= board - 1;
        count++;
    }
    return count;
}
