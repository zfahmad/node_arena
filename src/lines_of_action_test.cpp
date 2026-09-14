#include <constants.hpp>
#include <games/lines_of_action/lines_of_action.hpp>
#include <games/lines_of_action/lines_of_action_state.hpp>
#include <iostream>
#include <vector>

void print_bb_2(std::uint64_t bb, int num_rows) {
    std::uint64_t bit = 1ULL;
    std::cout << " ";
    for (int i = 0; i < num_rows; i++) {
        std::cout << GRAY << i << " ";
    }
    std::cout << RESET << "\n";
    for (int row = 0; row < num_rows; row++) {
        std::cout << " ";
        for (int col = 0; col < num_rows; col++) {
            if (bb & bit)
                std::cout << GREEN << CROSS << " " << RESET;
            else
                std::cout << GRAY << DOT << " " << RESET;
            bit = (bit << 1);
        }
        std::cout << "\n";
        bit = (bit << (ROW_MAX - num_rows));
    }
    std::cout << std::endl;
}

std::uint64_t bits_to_int(std::vector<std::uint8_t> bits) {
    std::uint64_t bb = 0ULL;
    for (std::uint8_t bit : bits)
        bb += (1ULL << bit);
    return bb;
}

int main(int argc, char **argv) {
    using BoardType = typename LinesOfActionState::BoardType;
    using BBType = typename LinesOfActionState::BBType;
    using Player = typename LinesOfActionState::Player;

    int num_rows = 5;
    // Setting board to initial config game
    LinesOfAction game{num_rows};
    LinesOfActionState state{num_rows};
    game.reset(state);
    
    // BBType bb_1 = 0x0000000100000000;
    // BBType bb_1 = 0x000000080C0E0A00;
    // BBType bb_1 = 0x000008000C0E0A00;
    // BBType bb_2 = 0x0000000000000000;
    // BBType bb_1 = 0x00000000000FF000;
    state.print_board();

    std::vector<LinesOfAction::ActionType> actions = game.get_actions(state);

    for (auto action : actions) {
        std::array<int, 3> inds = action_to_inds(action, num_rows);
        std::cout << action << "| " << inds[0] << "\n";
    }
    std::cout << std::endl;

    return 0;
}
