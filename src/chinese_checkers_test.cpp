#include <cstdint>
#include <games/chinese_checkers/chinese_checkers.hpp>
#include <games/chinese_checkers/chinese_checkers_state.hpp>
#include <iostream>

int main(int argc, char **argv) {
    std::cout << "Testing chinese_checkers...\n";
    ChineseCheckersState state = ChineseCheckersState(3, 3);
    ChineseCheckersState::BoardType board = ChineseCheckersState::BoardType({11, 416});
    state.set_board(board);
    state.print_board();

    // game.reset(state);
    state.print_board();
    // game.reset(small_state);
    // small_state.print_board();
    // game.reset(smaller_state);
    // smaller_state.print_board();

    // for ( auto shift : SHIFT_MASKS ) {
    //     ChineseCheckersState::BoardType bb = chinese_checkersState::BoardType({shift, 0});
    //     state.set_board(bb);
    //     state.print_board();
    // }
    // actions = game.get_actions(state);
    // std::cout << actions.size() << "\n";
    // // game.get_actions(small_state);
    // // game.get_actions(smaller_state);
    // for ( auto action : actions )
    //     std::cout << action << " ";
    // std::cout << std::endl;

    // state = game.get_next_state(state, actions[0]);
    // state.print_board();
    // actions = game.get_actions(state);
    // state = game.get_next_state(state, actions[1]);
    // state.print_board();
    // actions = game.get_actions(state);
    // state = game.get_next_state(state, actions[2]);
    // state.print_board();
    // actions = game.get_actions(state);
    // state = game.get_next_state(state, actions[2]);
    // state.print_board();

    // ChineseCheckers game = chinese_checkers();
    // std::vector<int> actions = game.get_actions(state);
    // for (int action : actions)
    //     std::cout << action << " ";
    // std::cout << std::endl;
    // game.apply_action(state, 1);
    // state.print_board();
    // game.undo_action(state, 1);
    // state.print_board();
    // state.set_board(ChineseCheckersState::BoardType({0, 0}));
    // game.apply_action(state, 0);
    // state.print_board();
    // game.undo_action(state, 0);
    // state.print_board();

    // ChineseCheckersState second_state = chinese_checkersState(5, 5);
    // bb1 = 0x000000000F000000;
    // bb2 = 0x00000003C0000000;
    // ChineseCheckersState::BoardType second_board = chinese_checkersState::BoardType({bb1, bb2});
    // second_state.set_board(second_board);
    // second_state.print_board();
    // std::cout << game.is_winner(second_state, ChineseCheckersState::Player::One) << "\n";
    // std::cout << game.is_winner(second_state, ChineseCheckersState::Player::Two) << std::endl;
    // bb1 = 0x00000000D070F0C0;
    // bb2 = 0x000000070F0D0700;
    // second_board = ChineseCheckersState::BoardType({bb1, bb2});
    // second_state.set_board(second_board);
    // second_state.print_board();
    // std::cout << game.is_draw(second_state) << std::endl;

    return 0;
}
