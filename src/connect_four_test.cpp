#include <cstdint>
#include <games/connect_four/connect_four.hpp>
#include <games/connect_four/connect_four_state.hpp>
#include <iostream>

int main(int argc, char **argv) {
    std::cout << "Testing connect four...\n";
    ConnectFourState state = ConnectFourState(4, 3);
    uint64_t bb1 = 0x0000000000030000;
    uint64_t bb2 = 0x0000000000041000;
    ConnectFourState::BoardType board = ConnectFourState::BoardType({bb1, bb2});
    state.set_board(board);
    // state.print_board();
    // std::string state_str = state.state_to_string();
    // state.set_board(ConnectFourState::BoardType({0, 0}));
    // state.print_board();
    // state.string_to_state(state_str);
    // state.print_board();
    //
    state.string_to_state("00690e71122e71000016710e6d550a00067");
    state.print_board();
    ConnectFour game = ConnectFour();
    std::cout << game.is_draw(state) << std::endl;
    // std::vector<int> actions = game.get_actions(state);
    // for (int action : actions)
    //     std::cout << action << " ";
    // std::cout << std::endl;
    // game.apply_action(state, 1);
    // state.print_board();
    // game.undo_action(state, 1);
    // state.print_board();
    // state.set_board(ConnectFourState::BoardType({0, 0}));
    // game.apply_action(state, 0);
    // state.print_board();
    // game.undo_action(state, 0);
    // state.print_board();

    // ConnectFourState second_state = ConnectFourState(5, 5);
    // bb1 = 0x000000000F000000;
    // bb2 = 0x00000003C0000000;
    // ConnectFourState::BoardType second_board = ConnectFourState::BoardType({bb1, bb2});
    // second_state.set_board(second_board);
    // second_state.print_board();
    // std::cout << game.is_winner(second_state, ConnectFourState::Player::One) << "\n";
    // std::cout << game.is_winner(second_state, ConnectFourState::Player::Two) << std::endl;
    // std::cout << game.shift_check(second_state.get_board()[ConnectFourState::Player::Two], 1) << std::endl;
    // bb1 = 0x00000000D070F0C0;
    // bb2 = 0x000000070F0D0700;
    // second_board = ConnectFourState::BoardType({bb1, bb2});
    // second_state.set_board(second_board);
    // state.print_board();
    // std::cout << static_cast<int>(game.get_outcome(state)) << std::endl;

    return 0;
}
