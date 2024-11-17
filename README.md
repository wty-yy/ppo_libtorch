## 使用方法
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

## 日志
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