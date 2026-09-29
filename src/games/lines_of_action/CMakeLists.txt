add_library(lines_of_action)

target_sources(lines_of_action PRIVATE lines_of_action_state.cpp lines_of_action.cpp)

target_include_directories(lines_of_action PUBLIC ${PROJECT_SOURCE_DIR}/include)

