#include <cstdint>
#include <games/tic_tac_toe/tic_tac_toe.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    TicTacToe::StateType game_state = TicTacToe::StateType();
    game_state.print_board();
    std::uint16_t bb0 = 0b0000000000111000;
    std::uint16_t bb1 = 0b0000000000000010;
    game_state.set_board(TicTacToe::StateType::BoardType({bb0, bb1}));
    game_state.print_board();
    std::cout << static_cast<int>(game_state.get_player()) << std::endl;
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
    for (const auto &action : actions)
        std::cout << action << " ";
    std::cout << std::endl;
    TicTacToe::StateType new_state = game.get_next_state(game_state, 4);
    new_state.print_board();
    new_state.string_to_state("2110122101");
    new_state.print_board();
    std::cout << game.is_winner(new_state, TicTacToe::StateType::Player::One) << "\n";
    std::cout << game.is_winner(new_state, TicTacToe::StateType::Player::Two)
              << std::endl;
    new_state.string_to_state("2211122120");
    new_state.print_board();
    std::cout << static_cast<int>(game.get_outcome(new_state)) << std::endl;
    std::cout << game.is_draw(new_state) << std::endl;
    std::cout << static_cast<int>(new_state.get_player()) << std::endl;
    new_state.string_to_state("2000201111");
    new_state.print_board();
    std::cout << static_cast<int>(game.get_outcome(new_state)) << std::endl;
    return 0;
}
