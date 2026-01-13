add_library(tic_tac_toe)

target_sources(tic_tac_toe PRIVATE tic_tac_toe.cpp tic_tac_toe_state.cpp)

set_target_properties(tic_tac_toe PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(tic_tac_toe PUBLIC ${PROJECT_SOURCE_DIR}/include)
