#include <cstdint>
#include <games/othello/othello.hpp>
#include <games/othello/othello_state.hpp>
#include <iostream>

int main(int argc, char **argv) {
    std::cout << "Testing othello...\n";
    OthelloState state = OthelloState(8, 8);
    OthelloState small_state = OthelloState(6, 6);
    OthelloState smaller_state = OthelloState(4, 4);
    Othello game = Othello();
    std::vector<Othello::ActionType> actions;

    game.reset(state);
    state.print_board();
    // game.reset(small_state);
    // small_state.print_board();
    // game.reset(smaller_state);
    // smaller_state.print_board();

    // for ( auto shift : SHIFT_MASKS ) {
    //     OthelloState::BoardType bb = OthelloState::BoardType({shift, 0});
    //     state.set_board(bb);
    //     state.print_board();
    // }
    actions = game.get_actions(state);
    std::cout << actions.size() << "\n";
    // game.get_actions(small_state);
    // game.get_actions(smaller_state);
    for ( auto action : actions )
        std::cout << action << " ";
    std::cout << std::endl;

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

    // Othello game = Othello();
    // std::vector<int> actions = game.get_actions(state);
    // for (int action : actions)
    //     std::cout << action << " ";
    // std::cout << std::endl;
    // game.apply_action(state, 1);
    // state.print_board();
    // game.undo_action(state, 1);
    // state.print_board();
    // state.set_board(OthelloState::BoardType({0, 0}));
    // game.apply_action(state, 0);
    // state.print_board();
    // game.undo_action(state, 0);
    // state.print_board();

    // OthelloState second_state = OthelloState(5, 5);
    // bb1 = 0x000000000F000000;
    // bb2 = 0x00000003C0000000;
    // OthelloState::BoardType second_board = OthelloState::BoardType({bb1, bb2});
    // second_state.set_board(second_board);
    // second_state.print_board();
    // std::cout << game.is_winner(second_state, OthelloState::Player::One) << "\n";
    // std::cout << game.is_winner(second_state, OthelloState::Player::Two) << std::endl;
    // bb1 = 0x00000000D070F0C0;
    // bb2 = 0x000000070F0D0700;
    // second_board = OthelloState::BoardType({bb1, bb2});
    // second_state.set_board(second_board);
    // second_state.print_board();
    // std::cout << game.is_draw(second_state) << std::endl;

    return 0;
}
