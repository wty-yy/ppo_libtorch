cmake_minimum_required(VERSION 3.10)
project(env_test VERSION 1.0)


set(EXECUTABLES
    train_ppo train_ppo.cpp
    eval_ppo eval_ppo.cpp
)

list(LENGTH EXECUTABLES EXECUTABLES_LENGTH)
math(EXPR NUM_TARGETS_MINUS_ONE "${EXECUTABLES_LENGTH} / 2 - 1")

foreach(i RANGE 0 ${NUM_TARGETS_MINUS_ONE})
  math(EXPR NAME_INDEX "${i} * 2")
  math(EXPR FILE_INDEX "${i} * 2 + 1")

  list(GET EXECUTABLES ${NAME_INDEX} EXEC_NAME)
  list(GET EXECUTABLES ${FILE_INDEX} SRC_FILE)

  add_executable(${EXEC_NAME} ${SRC_FILE})

  target_include_directories(${EXEC_NAME} PUBLIC env include)

  ### Torch ###
  find_package(Torch REQUIRED)
  target_link_libraries(${EXEC_NAME} PUBLIC "${TORCH_LIBRARIES}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set_property(TARGET ${EXEC_NAME} PROPERTY CXX_STANDARD 17)


  ### Tensorboard ###
  find_package(tensorboard_logger REQUIRED)
  target_link_libraries(${EXEC_NAME} PUBLIC ${TENSORBOARD_LOGGER_LIBRARIES} protobuf)

  ### Env Render ###
  find_package(SDL2 REQUIRED)
  target_link_libraries(${EXEC_NAME} PUBLIC ${SDL2_LIBRARIES})
  # target_link_libraries(${EXEC_NAME} PUBLIC sfml-graphics sfml-window sfml-system)

  add_compile_options(-O3)

endforeach()