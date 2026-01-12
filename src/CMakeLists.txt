add_executable(ttt_test main.cpp)
add_executable(c4_test connect_four_test.cpp)
add_executable(othello_test othello_test.cpp)

add_subdirectory(games)

target_link_libraries(ttt_test PRIVATE tic_tac_toe)
target_link_libraries(c4_test PRIVATE connect_four)
target_link_libraries(othello_test PRIVATE othello)

nanobind_add_module(tic_tac_toe_wrapper tic_tac_toe_wrapper.cpp)
set_target_properties(tic_tac_toe_wrapper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/wrapper
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/wrapper
)
target_link_libraries(tic_tac_toe_wrapper PRIVATE tic_tac_toe)
nanobind_add_stub(
    tic_tac_toe_wrapper_stub
    MODULE tic_tac_toe_wrapper
    PYTHON_PATH $<TARGET_FILE_DIR:tic_tac_toe_wrapper>
    DEPENDS tic_tac_toe_wrapper
    RECURSIVE
    OUTPUT tic_tac_toe_wrapper.pyi
)
