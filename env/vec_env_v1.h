#include "env.h"
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <future>

class VecEnv {
 public:
  VecEnv(std::function<std::unique_ptr<Env>()> envFactory, int numEnvs):numEnvs(numEnvs) {
    /*
    Args:
      envFactory: Env的子类的构造函数, 通过lambda函数直接构建所需的环境
      numEnvs: 并行环境数目
    */
    for (int i = 0; i < numEnvs; ++i) envs.emplace_back(envFactory());
  }

  std::vector<EnvInfo> reset() {
    std::vector<EnvInfo> results(numEnvs);
    runInParallel([this, &results](int i) {
      results[i] = envs[i]->reset();
    });
    return results;
  }

  std::vector<EnvInfo> step(const std::vector<int>& actions) {
    std::vector<EnvInfo> results(numEnvs);
    runInParallel([this, &results, &actions](int i) {
      results[i] = envs[i]->step(actions[i]);
      if (results[i].done) {  // 环境结束自动reset, 返回obs为reset后的obs
        EnvInfo info = envs[i]->reset();
        results[i].obs = info.obs;
      }
    });
    return results;
  }

 private:
  int numEnvs;
  std::vector<std::unique_ptr<Env>> envs;
  // 无并行: 执行1024*100步: 0.284000s
  // 并行执行100步:
  #if 1  // n_envs: time, 1024: 2.726000s, 4096: 10.835000s
  void runInParallel(const std::function<void(int)>& func) {
    /* 执行多线程并行处理函数func
    Args:
      func: 需并行处理的函数, 传入为环境编号(int) i, 返回为void
    */
    std::vector<std::thread> threads;
    for (int i = 0; i < numEnvs; ++i) threads.emplace_back(func, i);
    for (auto& thread : threads) if (thread.joinable()) thread.join();
  }
  #else  // n_envs: time, 1024: 2.896000s, 4096: 11.617000s
  void runInParallel(const std::function<void(int)>& func) {
    std::vector<std::future<void>> futures;  // 用于存储每个任务的 future

    // 异步启动任务，每个任务返回一个 std::future
    for (int i = 0; i < numEnvs; ++i) {
        futures.push_back(std::async(std::launch::async, func, i));  // 启动异步任务
    }

    // 等待所有任务完成
    for (auto& future : futures) {
        future.get();  // 等待任务完成
    }
  }
  #endif
};
