add_library(othello)

target_sources(othello PRIVATE othello.cpp othello_state.cpp)

target_include_directories(othello PUBLIC ${PROJECT_SOURCE_DIR}/include)
