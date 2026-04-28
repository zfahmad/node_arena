#include <cstdint>
#include <games/chinese_checkers/chinese_checkers.hpp>
#include <games/chinese_checkers/chinese_checkers_state.hpp>
#include <iostream>

int main(int argc, char **argv) {
    std::cout << "Testing chinese_checkers...\n";
    int dim = 5;
    int num_pieces = 3;
    ChineseCheckersState state = ChineseCheckersState(dim, dim, num_pieces);
    ChineseCheckers game = ChineseCheckers(dim, dim, num_pieces);
    // Initial board for 3x3x3
    // ChineseCheckersState::BoardType board =
    //     ChineseCheckersState::BoardType({132608, 201850880});
    // Initial board for 4x4x3
    // ChineseCheckersState::BoardType board =
    //     ChineseCheckersState::BoardType({132608, 103347650560});
    // state.set_board(board);
    // state.print_board();

    // std::uint8_t byte = 0;
    // std::cout << static_cast<int>(byte) << std::endl;
    // byte = ~byte;
    // std::cout << static_cast<int>(byte) << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     std::cout << SETUPS[i] << std::endl;
    //     ChineseCheckersState state = ChineseCheckersState(6, 6, i);
    //     ChineseCheckersState::BoardType board =
    //         ChineseCheckersState::BoardType({SETUPS[i], 0});
    //     game.reset(state);
    //     state.print_board();
    // }

    // state.print_mask(512);
    // game.print_mask(game.destinations_mask);
    // game.print_mask(game.empties_mask);
    game.reset(state);
    for (int i = 0; i < 3; i++)
        std::cout << state.piece_locations[0][i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < 3; i++)
        std::cout << state.piece_locations[1][i] << " ";
    std::cout << std::endl;
    ChineseCheckersState::BBType p1 = (1ULL << 13) + (1ULL << 19) + (1ULL << 26);
    ChineseCheckersState::BBType p2 = (1ULL << 33) + (1ULL << 35) + (1ULL << 41);
    // ChineseCheckersState::BBType p1 = (1ULL << 10) + (1ULL << 11) + (1ULL << 27);
    // ChineseCheckersState::BBType p2 = (1ULL << 20) + (1ULL << 36) + (1ULL << 43);
    state.set_board(ChineseCheckersState::BoardType({p1, p2}));
    for (int i = 0; i < 3; i++)
        std::cout << state.piece_locations[0][i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < 3; i++)
        std::cout << state.piece_locations[1][i] << " ";
    std::cout << std::endl;
    state.print_board();
    std::vector<ChineseCheckers::ActionType> actions = game.get_actions(state);
    for ( ChineseCheckers::ActionType a : actions ) {
        std::cout << a[0] << " -> " << a[1] << std::endl;
    }
    game.apply_action(state, actions[6]);
    state.print_board();

    // ChineseCheckers::StateType::BBType steps, hops;
    // steps = game.get_steps(state.get_board(), 11);
    // game.print_mask(steps);
    // steps = game.get_steps(state.get_board(), 25);
    // game.print_mask(steps);
    // steps = game.get_steps(state.get_board(), 52);
    // game.print_mask(steps);
    // steps = game.get_steps(state.get_board(), 20);
    // game.print_mask(steps);
    // hops = game.get_hops(state.get_board(), 20);
    // game.print_mask(hops);
    // steps = game.get_steps(state.get_board(), 36);
    // game.print_mask(steps);
    // hops = game.get_hops(state.get_board(), 36);
    // game.print_mask(hops);
    // steps = game.get_steps(state.get_board(), 43);
    // game.print_mask(steps);
    // hops = game.get_hops(state.get_board(), 43);
    // game.print_mask(hops);

    // game.print_mask(132608);
    // game.print_mask(103347650560);
    // std::cout << state.to_string() << std::endl;
    // state.print_board();
    // std::vector<std::vector<uint8_t>> state_array = state.to_array();
    //
    // for (auto arr : state_array) {
    //     for (int c : arr) {
    //         std::cout << c << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // game.reset(state);
    // state.print_board();
    // game.reset(small_state);
    // small_state.print_board();
    // game.reset(smaller_state);
    // smaller_state.print_board();

    // for ( auto shift : SHIFT_MASKS ) {
    //     ChineseCheckersState::BoardType bb =
    //     chinese_checkersState::BoardType({shift, 0}); state.set_board(bb);
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
    // ChineseCheckersState::BoardType second_board =
    // chinese_checkersState::BoardType({bb1, bb2});
    // second_state.set_board(second_board);
    // second_state.print_board();
    // std::cout << game.is_winner(second_state,
    // ChineseCheckersState::Player::One) << "\n"; std::cout <<
    // game.is_winner(second_state, ChineseCheckersState::Player::Two) <<
    // std::endl; bb1 = 0x00000000D070F0C0; bb2 = 0x000000070F0D0700;
    // second_board = ChineseCheckersState::BoardType({bb1, bb2});
    // second_state.set_board(second_board);
    // second_state.print_board();
    // std::cout << game.is_draw(second_state) << std::endl;

    return 0;
}
