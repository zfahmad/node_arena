import wrapper.tic_tac_toe_wrapper as gw

if __name__ == "__main__":
    print(gw.add(3, 5))
    state = gw.TicTacToeState()
    game = gw.TicTacToe()
    game.reset(state)
    state.print_board()
    actions = game.get_actions(state)
    print(actions)
    state = game.get_next_state(state, actions[4])
    state.print_board()
    actions = game.get_actions(state)
    print(actions)
