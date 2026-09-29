#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <constants.hpp>
#include <games/chinese_checkers/chinese_checkers_state.hpp>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using BoardType = ChineseCheckersState::BoardType;
using BBType = ChineseCheckersState::BBType;

ChineseCheckersState::ChineseCheckersState(int num_rows, int num_cols,
                                           int num_pieces) {
    if ((num_rows < 3) || (num_rows > 9) || (num_cols < 3) || (num_cols > 9)) {
        throw std::invalid_argument(
            "Board dimensions must be valid sizes between 3 and 9.");
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

    shift = MAX_ROW;
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
            bit = bit >> (MAX_ROW - 1);
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
            bit = bit >> (MAX_ROW - 1);
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int base_height(int num_pieces) {
    int height = (int)std::ceil((std::sqrt(1 + (MAX_ROW)*num_pieces) - 1) / 2);
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
        bits = bits >> (MAX_ROW + 1);
        for (int i = 0; i < num_rows_; i++) {
            for (int j = 0; j < num_cols_; j++) {
                player_arr.push_back(bits & 1);
                bits = bits >> 1;
            }
            bits = bits >> (MAX_ROW - num_rows_);
        }
        if (get_player() == Player::Two)
            std::reverse(player_arr.begin(), player_arr.end());
        arrs.push_back(std::move(player_arr));
    }
    return arrs;
}

std::vector<int> get_player_locations(ChineseCheckersState::BBType board) {
    std::vector<int> locations;
    ChineseCheckersState::BBType bit = (uint128_t)1;

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

std::vector<ChineseCheckersState::BBType>
ChineseCheckersState::to_compact() const {
    std::vector<BBType> board;
    board.reserve(2);
    board.push_back(board_[Player::One]);
    board.push_back(board_[Player::Two]);
    return board;
}

void ChineseCheckersState::from_compact(std::vector<BBType> compact_board) {
    if ((compact_board[0] & compact_board[1]) != 0)
        throw std::logic_error("Bit collision");
    board_[Player::One] = compact_board[0];
    board_[Player::Two] = compact_board[1];
}

std::string ChineseCheckersState::to_string() {
    // Converts the state representation to a string.
    // First sixteen characters represent the board for player one in hex.
    // First sixteen characters represent the board for player two in hex.
    // Last character is the current player at the state.
    std::string state_str = "";

    for (int i = 0; i < 2; i++) {
        state_str += std::to_string(piece_locations[i][0]);
        for (int j = 1; j < num_pieces_; j++)
            state_str += "," + std::to_string(piece_locations[i][j]);
        state_str += "|";
    }

    if (player_ == Player::One)
        state_str += "0";
    else
        state_str += "1";
    state_str +=
        std::to_string(this->num_rows_) + std::to_string(this->num_cols_);

    return state_str;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

void ChineseCheckersState::from_string(std::string state_str) {
    std::vector<std::string> parts = split(state_str, '|');
    BBType bb_1 = (uint128_t)0, bb_2 = (uint128_t)0;

    // int i = 0;
    for (auto loc : split(parts[0], ',')) {
        // piece_locations[0][i] = std::stoi(loc);
        bb_1 += (((uint128_t)1 << std::stoi(loc)));
        // i++;
    }

    // i = 0;
    for (auto loc : split(parts[1], ',')) {
        // piece_locations[1][i] = std::stoi(loc);
        bb_2 += ((uint128_t)1 << std::stoi(loc));
        // i++;
    }
    set_board(BoardType({bb_1, bb_2}));

    this->num_rows_ = parts[2][1] - '0';
    this->num_cols_ = parts[2][2] - '0';
    if (parts[2][0] == '0')
        set_player(Player::One);
    else
        set_player(Player::Two);
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

// BoardType ChineseCheckersState::reflect_vertical(BoardType board) {
//     BoardType new_board = board;
//     BoardType temp;
//
//     std::vector<Player> players;
//     players.push_back(Player::One);
//     players.push_back(Player::Two);
//
//     for (Player player : players) {
//         temp[player] = (new_board[player] ^ (new_board[player] >> 7)) &
//                        0x00AA00AA00AA00AAULL;
//         new_board[player] ^= temp[player] ^ (temp[player] << 7);
//         temp[player] = (new_board[player] ^ (new_board[player] >> 14)) &
//                        0x0000CCCC0000CCCCULL;
//         new_board[player] ^= temp[player] ^ (temp[player] << 14);
//         temp[player] = (new_board[player] ^ (new_board[player] >> 28)) &
//                        0x00000000F0F0F0F0ULL;
//         new_board[player] ^= temp[player] ^ (temp[player] << 28);
//     }
//
//     return new_board;
// }

BoardType ChineseCheckersState::reflect_vertical(BoardType board) {
    static constexpr int N = 11;
    BoardType new_board = board;
    std::vector<Player> players = {Player::One, Player::Two};

    for (Player player : players) {
        uint128_t src = board[player];
        uint128_t dst = 0;

        for (int r = 0; r < N; ++r) {
            uint128_t row = (src >> (r * N)) & (((uint128_t)1 << N) - 1);
            while (row) {
                int c = __builtin_ctzll(
                    (unsigned long long)row); // safe: row fits in 11 bits
                dst |= ((uint128_t)1 << (c * N + r));
                row &= row - 1;
            }
        }
        new_board[player] = dst;
    }
    return new_board;
}

BoardType ChineseCheckersState::flip_board(BoardType board) {
    // Rotates the board 180
    BoardType new_board = BoardType({(uint128_t)0, (uint128_t)0});
    BoardType temp = board;
    BBType bit = (uint128_t)1;

    std::vector<Player> players;
    players.push_back(Player::One);
    players.push_back(Player::Two);

    int offset = (9 - this->num_rows_);
    for (Player player : players) {
        temp[player] <<= (offset * 12);

        for (int i = 0; i < 121; i++) {
            new_board[player] |= (temp[player] & bit);
            temp[player] >>= 1;
            new_board[player] <<= 1;
        }
        new_board[player] >>= 1;
    }

    return BoardType({new_board[Player::Two], new_board[Player::One]});
}

std::array<BBType, 2> ChineseCheckersState::canonical_form() {
    std::vector<std::array<BBType, 2>> symmetries;
    BoardType board = get_board();
    BoardType transformed_board;

    if (this->player_ == Player::Two)
        board = flip_board(board);
    symmetries.push_back({board[Player::One], board[Player::Two]});

    transformed_board = reflect_vertical(board);
    symmetries.push_back({transformed_board[Player::One],
    transformed_board[Player::Two]});

    std::array<BBType, 2> canonical =
        *std::min_element(symmetries.begin(), symmetries.end());
    return canonical;
}

void ChineseCheckersState::from_canonical_form(
    std::array<BBType, 2> canonical_state) {
    // The canonical form of Chinese checkers states view states from the
    // perspective of the first player.
    // Loading a state from its canonical form also sets the first player as
    // acting.
    this->set_board(BoardType(canonical_state));
    this->set_player(Player::One);
}
