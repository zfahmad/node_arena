add_library(connect_four)

target_sources(connect_four PRIVATE connect_four.cpp connect_four_state.cpp)

target_include_directories(connect_four PUBLIC ${PROJECT_SOURCE_DIR}/include)
