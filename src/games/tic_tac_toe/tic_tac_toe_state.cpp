#include <cassert>
#include <constants.hpp>
#include <games/tic_tac_toe/tic_tac_toe_state.hpp>
#include <iostream>
#include <stdexcept>
#include <algorithm>

using BBType = TicTacToeState::BBType;
using BoardType = TicTacToeState::BoardType;

TicTacToeState::TicTacToeState(int num_rows, int num_cols) {
    this->num_rows_ = num_rows;
    this->num_cols_ = num_cols;
}

void TicTacToeState::print_board() {
    // Print the board to screen.
    // LSB of the bit representation is top-left cell;
    // MSB of the bit representation is bottom-right.
    for (int row = 0; row < 3; row++) {
        std::cout << GRAY << " " << (row * 3) << " ";
        for (int col = 0; col < 3; col++) {
            int bit = (row * 3) + col;
            if ((board_[Player::One] >> bit) & 1)
                std::cout << GREEN << CROSS << " ";
            else if ((board_[Player::Two] >> bit) & 1)
                std::cout << RED << NAUGHT << " ";
            else
                std::cout << RESET << DOT << " ";
        }
        std::cout << "\n";
    }
    std::cout << RESET << std::endl;
}

void TicTacToeState::set_board(BoardType board) { this->board_ = board; }

std::vector<TicTacToeState::BBType> TicTacToeState::to_compact() const {
    std::vector<BBType> board;
    board.reserve(3);
    board.push_back(board_[Player::One]);
    board.push_back(board_[Player::Two]);
    if (player_ == Player::One)
        board.push_back(0);
    else
        board.push_back(1);
    return board;
}

void TicTacToeState::from_compact(std::vector<BBType> compact_board) {
    if ((compact_board[0] & compact_board[1]) != 0)
        throw std::logic_error("Bit collision");
    board_[Player::One] = compact_board[0];
    board_[Player::Two] = compact_board[1];
    if (compact_board[2] == 0)
        player_ = Player::One;
    else
        player_ = Player::Two;
}

std::vector<std::vector<std::uint8_t>> TicTacToeState::to_array() {
    std::vector<std::vector<uint8_t>> arrs;
    arrs.reserve(2);
    std::vector<Player> players;
    players.push_back(get_player());
    players.push_back(get_opponent());

    for (Player p : players) {
        std::vector<uint8_t> player_arr;
        player_arr.reserve(9);
        BBType bits = board_[p];
        for (int i = 0; i < 9; i++)
            player_arr.push_back((bits >> i) & 1);
        arrs.push_back(std::move(player_arr));
    }

    return arrs;
}

std::string TicTacToeState::to_string() {
    // Converts the state representation to readable string.
    // First nine characters represent the board.
    // Last character is the current player at the state.
    std::string state_str = "";
    BoardType board = board_;
    for (int i = 0; i < 9; i++) {
        if (board[Player::One] & 1)
            state_str += '1';
        else if (board[Player::Two] & 1)
            state_str += '2';
        else
            state_str += '0';
        board[Player::One] = (board[Player::One] >> 1);
        board[Player::Two] = (board[Player::Two] >> 1);
    }
    if (player_ == Player::One)
        state_str += "0";
    else
        state_str += "1";
    return state_str;
}

void TicTacToeState::from_string(std::string state_str) {
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

BoardType TicTacToeState::rot_180(BoardType board) {
    BoardType new_board = BoardType({0, 0});
    std::vector<Player> players;
    players.push_back(Player::One);
    players.push_back(Player::Two);

    for (Player player : players){
        BBType left = 1;
        BBType right = 1 << 8;

        for (int i = 0; i < 9; i++) {
            if (board[player] & left) {
                new_board[player] |= right;
            }
            left <<= 1;
            right >>= 1;
        }
    }
    
    return new_board;
}

BoardType TicTacToeState::reflect_horizontal(BoardType board) {
    BBType reflect_mask = 0x0049;
    BoardType new_board = BoardType({0, 0});
    std::vector<Player> players;
    players.push_back(Player::One);
    players.push_back(Player::Two);

    for (Player player : players) {
        BBType pieces = reflect_mask & board[player];
        pieces = pieces << 2;
        new_board[player] |= pieces;
    }

    reflect_mask = 0x0092;

    for (Player player : players) {
        BBType pieces = reflect_mask & board[player];
        new_board[player] |= pieces;
    }

    reflect_mask = 0x0124;

    for (Player player : players) {
        BBType pieces = reflect_mask & board[player];
        pieces = pieces >> 2;
        new_board[player] |= pieces;
    }

    return new_board;
}

BoardType TicTacToeState::reflect_vertical(BoardType board) {
    // Vertical reflection is a 180 rotation followed by a horizontal
    // reflection.
    BoardType new_board = rot_180(board);
    new_board = reflect_horizontal(new_board);
    return new_board;
}

BBType swap_bit(BBType board, int pos_1, int pos_2) {
    BBType bit_1 = (board >> pos_1) & 1;
    BBType bit_2 = (board >> pos_2) & 1;
    BBType x = bit_1 ^ bit_2;
    BBType mask = (x << pos_1) | (x << pos_2);
    return board ^ mask;
}

BoardType TicTacToeState::reflect_diagonal_pos(BoardType board) {
    BoardType new_board = board;
    std::vector<Player> players;
    players.push_back(Player::One);
    players.push_back(Player::Two);

    for (Player player : players) {
        new_board[player] = swap_bit(board[player], 1, 3);
        new_board[player] = swap_bit(new_board[player], 2, 6);
        new_board[player] = swap_bit(new_board[player], 5, 7);
    }

    return new_board;
}

BoardType TicTacToeState::reflect_diagonal_neg(BoardType board) {
    BoardType new_board = board;
    std::vector<Player> players;
    players.push_back(Player::One);
    players.push_back(Player::Two);

    for (Player player : players) {
        new_board[player] = swap_bit(board[player], 1, 5);
        new_board[player] = swap_bit(new_board[player], 0, 8);
        new_board[player] = swap_bit(new_board[player], 3, 7);
    }

    return new_board;
}

BoardType TicTacToeState::rot_90(BoardType board) {
    // Reflection along the negative gradient plus a vertical reflection.
    
    BoardType new_board = reflect_diagonal_pos(board);
    new_board = reflect_horizontal(new_board);
    return new_board;
}

BoardType TicTacToeState::rot_270(BoardType board) {
    // Reflection along the positive gradient plus a vertical reflection.
    
    BoardType new_board = reflect_diagonal_neg(board);
    new_board = reflect_horizontal(new_board);
    return new_board;
}

std::array<BBType, 2> TicTacToeState::canonical_form() {
    std::vector<std::array<BBType, 2>> symmetries;
    BoardType board = get_board();
    BoardType transformed_board;
    symmetries.push_back({board[Player::One], board[Player::Two]});

    transformed_board = rot_90(board);
    symmetries.push_back({transformed_board[Player::One], transformed_board[Player::Two]});
    transformed_board = rot_180(board);
    symmetries.push_back({transformed_board[Player::One], transformed_board[Player::Two]});
    transformed_board = rot_270(board);
    symmetries.push_back({transformed_board[Player::One], transformed_board[Player::Two]});
    transformed_board = reflect_horizontal(board);
    symmetries.push_back({transformed_board[Player::One], transformed_board[Player::Two]});
    transformed_board = reflect_vertical(board);
    symmetries.push_back({transformed_board[Player::One], transformed_board[Player::Two]});
    transformed_board = reflect_diagonal_pos(board);
    symmetries.push_back({transformed_board[Player::One], transformed_board[Player::Two]});
    transformed_board = reflect_diagonal_neg(board);
    symmetries.push_back({transformed_board[Player::One], transformed_board[Player::Two]});

    std::array<BBType, 2> canonical = *std::min_element(symmetries.begin(), symmetries.end());
    return canonical;
}
