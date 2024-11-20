# 基于libtorch使用纯C++下的PPO算法训练贪吃蛇游戏
## 目录
- [效果图](#效果图)
- [使用方法](#使用方法)
    - [依赖包](#依赖包)
    - [VSCode配置](#vscode配置)
- [更新日志](#更新日志)
- [调参](#调参)
## 效果图
|![8x8训练4h](./assets/snake8x8.gif)|![8x8训练4h](./assets/snake8x8.gif)|
|-|-|
|8x8大小训练4h(7e8步)|-|
## 使用方法
### 依赖包
1. [GitHub - summary_writer](https://github.com/wty-yy/summary_writer): 用于绘制TensorBoard曲线
2. SDL2下载: `sudo apt install libsdl2-dev`

### VSCode配置
`.vscode/settings.json`中加入
```json
"cmake.configureArgs": [
  "-DCMAKE_PREFIX_PATH=/path/to/libtorch",  // ex: /home/yy/lib/libtorch-2.5.1+cu121
  "-DCMAKE_CUDA_COMPILER=/path/to/nvcc",  // ex: /usr/local/cuda/bin/nvcc
  "-DCAFFE2_USE_CUDNN=True"
]
```
(可选)`ctrl+shift+p`找到`C/C++: Edit Configurations (UI)`中包含路径，其中加入
```json
/path/to/libtorch/include/torch/csrc/api/include
/path/to/libtorch/include
```
使用CMake Tools编译运行即可

## 更新日志
### 2024.11.11.
完成环境制作`env.h, env_snake.h`: 支持渲染, 键盘游玩选项
### 2024.11.12.
支持多进程环境`VecEnv`: 当某一个环境done时, 自动reset, 返回的obs替换成reset后的obs

没有考虑到step到达的终止状态中，蛇头可能越界，从而导致获取obs数组时越界(只有多线程中才发现此问题)，修改为如果done则返回obs全零，因为也会被reset的覆盖掉。

发现写出来的`VecEnv`速度比单线程速度还满，主要原因是每次创建一个新的进程，创建进程效率过低，因此需要为每个环境单独维护一个持续运行的进程
### 2024.11.13.
实现`VecEnv`的难点主要是线程锁和阻塞，使用
1. 区域上锁: 
  ```cpp
  {
    std::lock_guard<std::mutex> lock(cmd_mutex);  // 保护一次命令相关变量的原子性}
    ...  // 上锁代码
  }
  ```
2. 区域上锁后阻塞:
  ```cpp
  std::unique_lock<std::mutex> lock(cmd_mutex);
  cmd_cv.wait(lock, [this](){return cmd != NONE;});  // wait时会自动释放lock锁, 直到被notify_one唤醒, 重新获得lock锁
  ```
3. 修改变量后通知放过阻塞:
  ```cpp
  {
    std::lock_guard<std::mutex> lock(info_mutex);
    info_ready = true;
  }
  info_cv.notify_one();
  ```
实现对`step, reset, 析构函数`的阻塞与信息处理, 每步之间都通过阻塞完成: `submit_action -> run -> get_result`, 第一个和最后一个由主进程调用, 中间run为待机进程, 当收到启动信号(`STEP, RESET, STOP`)开始计算.

## 调参
贪吃蛇为离散奖励任务，尤其是后期场景的决策与开始时差别很大因此需要大量训练，尝试了以下不同的奖励配置：

记重置游戏(撞墙或者吃到自己)的奖励为`reward_done`, 每走一步的奖励为`reward_step`, 每吃到一个食物的奖励为`reward_food`, 默认的`lr=2.5e-4, ent_coef=1e-2` (参考cleanrl_ppo的超参数)
1. v1: `reward_down=-1, reward_step=+1`, 可以训练出吃到51次(5e8 step)食物的模型, 但是非常不稳定（可能是没有动态调整学习率和熵系数）
  - 尝试1: `reward_step=-0.01`训练开始直接崩溃
  - 尝试2: `reward_down=-5 or -10, reward_step=+0.2`训练开始直接崩溃
2. v2: 对学习率和熵系数的动态调整, 删除重置游戏时的奖励: `reward_down=0, reward_step=0.2`, `lr: 2.5e-4 -> 1e-5, ent_coef: 0.01 -> 1e-5`, 比较稳定但是最好只能吃到42个(5e8 step)
3. v3:
  - 对食物奖励动态调整, 约到后期吃到的食物对应的奖励越高, 记当前是第`score`个食物, 则对应的奖励为`max(0.1, score/(width*height))`, 即按照比例进行分配奖励且最少为0.1分
  - 对动态调整的终点略微提高些: `lr: 2.5e-4 -> 5e-5, ent: 0.01 -> 5e-4`
  - 训练效果: 5e8能稳定达到50分以上, 最高52分; 接着训练到1e9, 仍然没有提升
4. v4:
  - 调参: 调整`gamma: 0.95 -> 0.995`, 将最终收敛的学习率进一步降低些 `lr: 2.5e-4 -> 2e-5, ent: 0.01 -> 1e-5`
  - 训练效果: 训练7.5e8步总算收敛, 达到最优策略, 得到64分
5. v5:
  - 调整总训练步数: `total_steps: 8e8, anneal_steps: 7e8`