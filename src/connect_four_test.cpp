#include <cstdint>
#include <games/connect_four/connect_four.hpp>
#include <games/connect_four/connect_four_state.hpp>
#include <iostream>

int main(int argc, char **argv) {
    std::cout << "Testing connect four...\n";
    ConnectFourState state = ConnectFourState(4, 3);
    uint64_t bb1 = 0x0000000000003000;
    uint64_t bb2 = 0x0000000000004200;
    ConnectFourState::BoardType board = ConnectFourState::BoardType({bb1, bb2});
    state.set_board(board);
    state.print_board();
    std::string state_str = state.state_to_string();
    state.set_board(ConnectFourState::BoardType({0, 0}));
    state.print_board();
    state.string_to_state(state_str);
    state.print_board();

    ConnectFour game = ConnectFour();
    std::vector<int> actions = game.get_actions(state);
    for (int action : actions)
        std::cout << action << " ";
    std::cout << std::endl;
    game.apply_action(state, 1);
    state.print_board();
    game.undo_action(state, 1);
    state.print_board();
    state.set_board(ConnectFourState::BoardType({0, 0}));
    game.apply_action(state, 0);
    state.print_board();
    game.undo_action(state, 0);
    state.print_board();

    return 0;
}
