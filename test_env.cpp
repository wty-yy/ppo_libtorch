#include "env_snake.h"
#include "vec_env.h"
#include <chrono>
#include <thread>

void start_single_game(SnakeGameOption option={}, int total_steps=1000) {
  SnakeGame game(option);
  EnvInfo info = game.reset();
  // for (int i = 0; i < total_steps; i++) {
  //   printf("Step: %d\n", i+1);
  //   int action = rand() % 4;
  //   EnvInfo info = game.step(action);
  //   if (info.done) game.reset();
  //   // if (option.useRender()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  //   if (option.useRender()) SDL_Delay(10);
  // }
  if (option._useRender) game.play();
}

/*
process: 1 time used: 3.097000s
process: 2 time used: 1.626000s
process: 4 time used: 0.908000s
process: 8 time used: 0.562000s
process: 16 time used: 0.404000s
process: 32 time used: 0.330000s
process: 64 time used: 0.315000s
process: 128 time used: 0.332000s
process: 256 time used: 0.337000s
process: 512 time used: 0.363000s
process: 1024 time used: 0.479000s
process: 2048 time used: 0.634000s
process: 4096 time used: 0.811000s
time used: 10.205000s
*/
void start_venv_game(int num_envs, int total_steps=100, bool verbose=false, SnakeGameOption &&option={}) {
  auto venv = VecEnv([option](int i){return std::make_shared<SnakeGame>(option);}, num_envs);
  auto infos = venv.reset();
  for (int i = 0; i < total_steps; ++i) {
    if (verbose) printf("Step: %d\n", i+1);
    std::vector<int> actions(num_envs, 0);
    for (int j = 0; j < num_envs; ++j) actions[j] = rand() % 4;
    auto infos = venv.step(actions);
    // for (int j = 0; j < num_envs; ++j) if (infos[j].done) printf("Done: id=%d \n", j);
  }
}

int main() {
  srand(42);
  printf("rand:%d\n", rand());
  auto start_time = std::chrono::high_resolution_clock::now();
  // start_single_game(false, 1024*4*100);  // 1024*4*100用时2.325s, 1024*8*100用时4.676s
  start_single_game(SnakeGameOption().seed(NAN).useRender(true).width(8).height(8).gridSize(80));
  // start_venv_game(64, 1e6/64);  // 64线程, 1e6: 0.945s, 1e8: 80s
  #if 0  // 测试不同线程下速度
  for (int num_process = 1; num_process <= 4096; num_process *= 2) {
    auto t1 = std::chrono::high_resolution_clock::now();
    start_venv_game(num_process, 1024*4*100/num_process);  // 64线程最快, 1024*4*100用时0.315s
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    printf("process: %d time used: %lfs\n", num_process, 1.0*t3.count()/1000);
  }
  #endif
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  printf("time used: %lfs\n", 1.0*duration.count()/1000);
  return 0;
}
