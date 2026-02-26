add_library(chinese_checkers)

target_sources(chinese_checkers PRIVATE chinese_checkers.cpp chinese_checkers_state.cpp)

target_include_directories(chinese_checkers PUBLIC ${PROJECT_SOURCE_DIR}/include)
