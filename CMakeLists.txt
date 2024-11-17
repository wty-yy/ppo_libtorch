cmake_minimum_required(VERSION 3.10)
project(env_test VERSION 1.0)


# set(RUN test_torch)
# set(RUN train_ppo)
set(RUN eval_ppo)
# set(RUN test_env)
set(CPP_STEM ${RUN})

add_executable(${RUN} ${CPP_STEM}.cpp)

target_include_directories(${RUN} PUBLIC env include)

### Torch ###
find_package(Torch REQUIRED)
target_link_libraries(${RUN} PUBLIC "${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set_property(TARGET ${RUN} PROPERTY CXX_STANDARD 17)


### Tensorboard ###
find_package(tensorboard_logger REQUIRED)
target_link_libraries(${RUN} PUBLIC ${TENSORBOARD_LOGGER_LIBRARIES} protobuf)

### Env Render ###
find_package(SDL2 REQUIRED)
target_link_libraries(${RUN} PUBLIC ${SDL2_LIBRARIES})
# target_link_libraries(${RUN} PUBLIC sfml-graphics sfml-window sfml-system)
