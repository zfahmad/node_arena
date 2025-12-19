#ifndef STATE_H
#define STATE_H

#include <concepts>
#include <string>

template <typename S, typename BoardType>
concept State =
    requires(S s, BoardType board, const std::string &state_str, S::Player player) {
        // State member access functions
        // print_board illustrates the board as we would see it when playing.
        // Get board should return an array of bitboards consisting of a
        // bitboard for each player.
        { s.print_board() } -> std::same_as<void>;
        { s.get_board() } -> std::same_as<const BoardType &>;
        { s.set_board(board) } -> std::same_as<void>;
        { s.get_player() } -> std::same_as<typename S::Player>;
        { s.set_player(player) } -> std::same_as<void>;

        // Functions used for converting from a state to a printable string.
        // Needed for logging games and loading games.
        { s.state_to_string() } -> std::same_as<std::string>;
        { s.string_to_state(state_str) } -> std::same_as<void>;
    };

#endif
