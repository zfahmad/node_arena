#ifndef CHINESE_CHECKERS_GENERATOR_HPP
#define CHINESE_CHECKERS_GENERATOR_HPP

#include <games/chinese_checkers/chinese_checkers.hpp>

class Generator {
public:
    Generator() {};
    std::vector<std::array<ChineseCheckersState::BBType, 2>>
    generate_all_states(const ChineseCheckers &game,
                        ChineseCheckersState &state);
    void generate_terminal_states(const ChineseCheckers &game,
                                  ChineseCheckersState &state);
};

#endif
