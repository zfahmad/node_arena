add_executable(ttt_test main.cpp)
add_executable(c4_test connect_four_test.cpp)
add_executable(othello_test othello_test.cpp)

add_subdirectory(games)

target_link_libraries(ttt_test PRIVATE tic_tac_toe)
target_link_libraries(c4_test PRIVATE connect_four)
target_link_libraries(othello_test PRIVATE othello)
