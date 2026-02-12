#ifndef TIC_TAC_TOE_HPP
#define TIC_TAC_TOE_HPP

#include <game.hpp>
#include <games/tic_tac_toe/tic_tac_toe_state.hpp>
#include <vector>
#include <string>

class TicTacToe {
public:
    enum class Outcomes { NonTerminal, P1Win, P2Win, Draw };
    using ActionType = int;
    using StateType = TicTacToeState;

    TicTacToe();
    std::string get_id() { return "tic_tac_toe"; }
    void reset(StateType &state);
    std::vector<ActionType> get_actions(const StateType &state) const;
    int apply_action(StateType &state, ActionType action);
    int undo_action(StateType &state, ActionType action);
    StateType get_next_state(const StateType &state, ActionType action);
    bool is_winner(const StateType &state, StateType::Player player);
    bool is_draw(const StateType &state);
    bool is_terminal(const StateType &state);
    Outcomes get_outcome(const StateType &state);
    std::vector<std::uint8_t> legal_moves_mask(const StateType &state);
};

static_assert(Game<TicTacToe>);

#endif
