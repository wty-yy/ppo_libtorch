#pragma once

#include <vector>
#include <memory>

struct EnvInfo {
  EnvInfo(){}
  EnvInfo(std::shared_ptr<std::vector<float>> obs, double reward, bool done):obs(obs), reward(reward), done(done){}
  std::shared_ptr<std::vector<float>> obs;
  double reward;
  bool done;
};

struct Env {
 public:
  Env(int obs_space, int action_nums):obs_space(obs_space), action_nums(action_nums){}
  virtual EnvInfo step(int) = 0;  // 离散动作空间
  virtual EnvInfo reset() = 0;
  virtual ~Env() = default;
  std::pair<int, int> get_space() {
    return {obs_space, action_nums};
  }
 protected:
  int obs_space, action_nums;
};
