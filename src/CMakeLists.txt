add_executable(ttt_test main.cpp)
add_executable(c4_test connect_four_test.cpp)
add_executable(othello_test othello_test.cpp)

add_subdirectory(games)

target_link_libraries(ttt_test PRIVATE tic_tac_toe)
target_link_libraries(c4_test PRIVATE connect_four)
target_link_libraries(othello_test PRIVATE othello)

set(GAMES
    tic_tac_toe
    connect_four
)

foreach(lib IN LISTS GAMES)
    nanobind_add_module(${lib}_wrapper ${lib}_wrapper.cpp)
    set_target_properties(${lib}_wrapper PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/wrappers
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/wrappers
    )
    target_link_libraries(${lib}_wrapper PRIVATE ${lib})
    nanobind_add_stub(
        ${lib}_wrapper_stub
        MODULE ${lib}_wrapper
        PYTHON_PATH $<TARGET_FILE_DIR:${lib}_wrapper>
        DEPENDS ${lib}_wrapper
        RECURSIVE
        OUTPUT ${lib}_wrapper.pyi
    )
endforeach()
