#include "tic_tac_toe_state.h"
#include "tic_tac_toe.h"
#include <iostream>
#include <cstdint>

int main(int argc, char* argv[]) {
    TicTacToeState game_state = TicTacToeState();
    game_state.print_board();
    std::uint16_t bb0 = 0b0000000000111000;
    std::uint16_t bb1 = 0b0000000000000010;
    game_state.set_board({bb0, bb1});
    game_state.print_board();
    std::cout << static_cast<int>(static_cast<int>(game_state.get_player())) << std::endl;
    int a = static_cast<int>(game_state.get_player());
    a = 8;
    std::cout << a << std::endl;
    game_state.print_board();
    std::cout << game_state.state_to_string() << std::endl;
    std::string state_str = "2010122100";
    game_state.string_to_state(state_str);
    game_state.print_board();
    std::cout << static_cast<int>(game_state.get_player()) << std::endl;

    TicTacToe game = TicTacToe();
    game.reset(game_state);
    game_state.print_board();
    std::cout << static_cast<int>(game_state.get_player()) << std::endl;
    std::vector<TicTacToe::ActionType> actions = game.get_actions(game_state);
    for( const auto &action : actions )
        std::cout << action << " ";
    std::cout << std::endl;
    return 0;
}
