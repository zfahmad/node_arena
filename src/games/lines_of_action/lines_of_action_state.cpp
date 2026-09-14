#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <constants.hpp>
#include <games/lines_of_action/lines_of_action_state.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>


using BoardType = LinesOfActionState::BoardType;
using BBType = LinesOfActionState::BBType;

void LinesOfActionState::print_board() {
    BBType bit = 1ULL;

    for (int i = 0; i < num_rows_; i++) {
        std::cout << " ";
        for (int j = 0; j < num_rows_; j++) {
            if (board_[Player::One] & bit)
                std::cout << BLUE << MULTIPLIER << " " << RESET;
            else if (board_[Player::Two] & bit)
                std::cout << RED << NAUGHT << " " << RESET;
            else
                std::cout << GRAY << CIRCLE << " " << RESET;
            bit = bit << 1;
        }
        bit = bit << (ROW_MAX - num_rows_);
        std::cout << "\n";
    }
    std::cout << std::endl;
}

std::vector<std::vector<uint8_t>> LinesOfActionState::to_array() {
    std::vector<std::vector<uint8_t>> arrs;
    arrs.reserve(2);
    std::vector<Player> players;
    players.push_back(get_player());
    players.push_back(get_opponent());

    for (Player p : players) {
        std::vector<uint8_t> player_arr;
        player_arr.reserve(num_rows_ * num_rows_);
        BBType bits = board_[p];
        for (int i = 0; i < num_rows_; i++) {
            for (int j = 0; j < num_rows_; j++) {
                player_arr.push_back(bits & 1);
                bits = bits >> 1;
            }
            bits = bits >> (ROW_MAX - num_rows_);
        }
        arrs.push_back(std::move(player_arr));
    }
    return arrs;
}

void LinesOfActionState::set_board(BoardType board) { this->board_ = board; }

std::vector<LinesOfActionState::BBType> LinesOfActionState::to_compact() const {
    std::vector<BBType> board;
    board.reserve(2);
    board.push_back(board_[Player::One]);
    board.push_back(board_[Player::Two]);
    return board;
}

void LinesOfActionState::from_compact(std::vector<BBType> compact_board) {
    if ((compact_board[0] & compact_board[1]) != 0)
        throw std::logic_error("Bit collision");
    board_[Player::One] = compact_board[0];
    board_[Player::Two] = compact_board[1];
}

std::string LinesOfActionState::to_string() {
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
    state_str += std::to_string(this->num_rows_);

    return state_str;
}

void LinesOfActionState::from_string(std::string state_str) {
    const char *data = state_str.data();
    auto r1 = std::from_chars(data, data + 16, board_[Player::One], 16);
    auto r2 = std::from_chars(data + 16, data + 32, board_[Player::Two], 16);

    assert((!(board_[Player::One] & board_[Player::Two])) &&
           "State has overlapping pieces.");
    int p1_num_pieces = num_pieces(board_[Player::One]);
    assert(p1_num_pieces == num_pieces(board_[Player::Two]));
    this->num_rows_ = state_str[33] - '0';

    if (state_str[32] == '0')
        set_player(Player::One);
    else
        set_player(Player::Two);
}

int LinesOfActionState::num_pieces(BBType board) const {
    // Counts the number of pieces given a specific player's board.
    int count = 0;
    while (board) {
        board &= board - 1;
        count++;
    }
    return count;
}

std::array<BBType, 2> LinesOfActionState::canonical_form() {
    std::array<BBType, 2> canonical_form;
    return canonical_form;
}

void LinesOfActionState::from_canonical_form(
    std::array<BBType, 2> canonical_state) {
    // The canonical form of Chinese checkers states view states from the
    // perspective of the first player.
    // Loading a state from its canonical form also sets the first player as
    // acting.
    this->set_board(BoardType(canonical_state));
    this->set_player(Player::One);
}
