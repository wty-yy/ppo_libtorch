#pragma once

#include "env.h"
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <condition_variable>
#include <mutex>

enum base_cmd {NONE, STEP, RESET, STOP};

class WorkerThread {
 public:
  WorkerThread(std::shared_ptr<Env> env):
    env(env), action(-1), cmd(NONE) {
    worker_thread = std::thread(&WorkerThread::run, this);
  }

  ~WorkerThread() {
    {
      std::lock_guard<std::mutex> lock(cmd_mutex);
      cmd = STOP;
    }
    cmd_cv.notify_one();
    if (worker_thread.joinable()) worker_thread.join();
  }
  
  void submit_action(int action) {
    {
      std::lock_guard<std::mutex> lock(cmd_mutex);  // 保护一次命令相关变量的原子性
      cmd = STEP;
      this->action = action;
    }
    cmd_cv.notify_one();  // 唤醒condition.wait
  }

  void submit_reset() {
    {
      std::lock_guard<std::mutex> lock(cmd_mutex);
      cmd = RESET;
    }
    cmd_cv.notify_one();  // 唤醒condition.wait
  }

  EnvInfo get_info() {
    std::unique_lock<std::mutex> lock(info_mutex);
    info_cv.wait(lock, [this](){return info_ready;});  // 等待run完成一步推理
    info_ready = false;
    return info;
  }

 private:
  std::shared_ptr<Env> env;
  std::thread worker_thread;
  std::mutex cmd_mutex;
  std::mutex info_mutex;
  std::condition_variable cmd_cv;
  std::condition_variable info_cv;
  base_cmd cmd;
  int action;
  EnvInfo info;
  bool info_ready;

  void run() {
    while (true) {
      {  // 设置unique_lock的作用范围, 保护cmd原子性, 执行本次命令不会执行后续的命令
        std::unique_lock<std::mutex> lock(cmd_mutex);
        cmd_cv.wait(lock, [this](){return cmd != NONE;});  // wait时会自动释放lock锁, 直到被notify_one唤醒, 重新获得lock锁
        if (cmd == STEP) {
          // printf("STEP\n");
          info = env->step(action);
        } else if (cmd == RESET) {
          // printf("RESET\n");
          info = env->reset();
        } else if (cmd == STOP) {
          // printf("STOP\n");
          break;
        }
        cmd = NONE;
      }
      if (info.done) { // 如果环境done, 则将obs更新为reset后的obs
        info.obs = env->reset().obs;
      }
      {
        std::lock_guard<std::mutex> lock(info_mutex);
        info_ready = true;
      }
      info_cv.notify_one();
    }
  }
};

class VecEnv {
 public:
  VecEnv(std::function<std::shared_ptr<Env>(int i)> envFactory, int numEnvs):numEnvs(numEnvs) {
    /*
    Args:
      envFactory: Env的子类的构造函数, 通过lambda函数直接构建所需的环境
      numEnvs: 并行环境数目
    */
    for (int i = 0; i < numEnvs; ++i) {
      envs.emplace_back(envFactory(i));
      // WorkerThread tmp(envs[i]);
      workers.emplace_back(std::make_unique<WorkerThread>(envs[i]));
    }
  }

  std::vector<EnvInfo> reset() {
    std::vector<EnvInfo> infos(numEnvs);
    for (int i = 0; i < numEnvs; ++i) workers[i]->submit_reset();
    for (int i = 0; i < numEnvs; ++i) infos[i] = workers[i]->get_info();
    return infos;
  }

  std::vector<EnvInfo> step(const std::vector<int>& actions) {
    std::vector<EnvInfo> infos(numEnvs);
    for (int i = 0; i < numEnvs; ++i) workers[i]->submit_action(actions[i]);
    for (int i = 0; i < numEnvs; ++i) infos[i] = workers[i]->get_info();
    return infos;
  }

  std::pair<int, int> get_space() { return envs[0]->get_space(); }

 private:
  int numEnvs;
  std::vector<std::shared_ptr<Env>> envs;
  std::vector<std::unique_ptr<WorkerThread>> workers;
};
