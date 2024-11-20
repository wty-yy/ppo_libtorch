#pragma once

#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <memory>
#include <cassert>
#include "env.h"
#include <cmath>
#include <random>


enum Direction { Up, Down, Left, Right };

struct SnakeSegment {
  int x, y;
  SnakeSegment(int x, int y):x(x), y(y) {}
};

struct SnakeGameOption {
  double _seed;
  int _width, _height, _gridSize, _windowWidth, _windowHeight;
  bool _useRender;
  double _reward_step, _reward_done, _reward_food;

  SnakeGameOption(
  double seed=NAN, bool useRender=false, int width=16, int height=16,
  int gridSize=40, double reward_step=-0.01, double reward_done=-10, double reward_food=1) : 
    _seed(seed), _useRender(useRender), _width(width), _height(height), _gridSize(gridSize),
    _reward_step(reward_step), _reward_done(reward_done), _reward_food(reward_food) {
    _windowWidth = _width * _gridSize;
    _windowHeight = _height * _gridSize;
  }
  SnakeGameOption &&seed(double x) {_seed = x; return std::move(*this);}
  SnakeGameOption &&useRender(bool x) {_useRender = x; return std::move(*this);}
  SnakeGameOption &&width(int x) {_width = x; update(); return std::move(*this);}
  SnakeGameOption &&height(int x) {_height = x; update(); return std::move(*this);}
  SnakeGameOption &&gridSize(int x) {_gridSize = x; update(); return std::move(*this);}
  SnakeGameOption &&reward_step(double x) {_reward_step = x; return std::move(*this);}
  SnakeGameOption &&reward_done(double x) {_reward_done = x; return std::move(*this);}
  SnakeGameOption &&reward_food(double x) {_reward_food = x; return std::move(*this);}

  void update() {
    _windowWidth = _width * _gridSize;
    _windowHeight = _height * _gridSize;
  }
};

class SnakeGame: public Env{
 public:
  SnakeGame(SnakeGameOption option) :
  Env(option._width * option._height, 4),
  seed(option._seed), rng(option._seed),
  useRender(option._useRender),
  width(option._width),
  height(option._height),
  gridSize(option._gridSize),
  reward_step(option._reward_step),
  reward_done(option._reward_done),
  reward_food(option._reward_food),
  score(0), deadCount(0), foodEaten(false) {
    printf("seed=%.0lf\n", seed);
    if (std::isnan(seed)) rng = std::mt19937(time(NULL));
    if (useRender) {
      if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        exit(1);
      }
      window = SDL_CreateWindow("Snake", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, option._windowWidth, option._windowHeight, SDL_WINDOW_SHOWN);
      if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        exit(1);
      }
      renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
      if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        exit(1);
      }
    }
    reset();
  }

  ~SnakeGame() {
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
  }

  EnvInfo step(int action) {
    assert(!done);
    if (useRender) handleInput(false);
    switch (action) {
      case Up: if (direction != Down) direction = Up; break;
      case Down: if (direction != Up) direction = Down; break;
      case Left: if (direction != Right) direction = Left; break;
      case Right: if (direction != Left) direction = Right; break;
      default: break;
    }
    update();
    return get_step_info();
  }

  EnvInfo reset() {
    if (useRender && score != 0) printf("episode score: %d\n", score);
    score = 0;
    reward = 0;
    done = false;
    snake.clear();
    snake.emplace_back(width / 2, height / 2);
    std::uniform_int_distribution<int> dist(0, Right);
    direction = static_cast<Direction>(dist(rng));
    // printf("direction:%d\n", direction);
    generateFood();
    // if (useRender) render();  // segment error?
    return get_step_info();
  }

  void play() {
    if (!useRender) {
      printf("Render MODE hasn't been started! Can't play...Exit\n");
      exit(0);
    }
    while (true) {
      handleInput(true);
      update();
      if (done) reset();
      else render();
      SDL_Delay(100);  // Sleep for 100ms to control game speed
    }
  }

  int get_game_size() { return std::min(width, height); }

 private:
  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;
  std::vector<SnakeSegment> snake;
  Direction direction;
  SDL_Point food;
  bool foodEaten, done, useRender;
  int score, deadCount;
  int width, height, gridSize;
  double seed, reward;
  double reward_step, reward_food, reward_done;
  std::mt19937 rng;

  void handleInput(bool humanPlay=false) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        done = true;
        SDL_DestroyWindow(window);
        SDL_Quit();
      }
      if (humanPlay && event.type == SDL_KEYDOWN) {
        switch (event.key.keysym.sym) {
          case SDLK_UP: if (direction != Down) direction = Up; break;
          case SDLK_DOWN: if (direction != Up) direction = Down; break;
          case SDLK_LEFT: if (direction != Right) direction = Left; break;
          case SDLK_RIGHT: if (direction != Left) direction = Right; break;
          default: break;
        }
      }
    }
  }

  void update() {
    reward = reward_step;
    moveSnake();
    checkCollisions();
    if (foodEaten) {
      snake.push_back(snake.back());
      foodEaten = false;
      generateFood();
    }
  }

  void moveSnake() {
    for (int i = snake.size() - 1; i > 0; --i) {
      snake[i] = snake[i - 1];
    }
    switch (direction) {
      case Up: --snake[0].y; break;
      case Down: ++snake[0].y; break;
      case Left: --snake[0].x; break;
      case Right: ++snake[0].x; break;
    }
  }

  void checkCollisions() {
    // Check if snake eats the food
    if (snake[0].x == food.x && snake[0].y == food.y) {
      foodEaten = true;
      score += 1;
      reward = reward_food;
      if (std::isnan(reward)) {
        reward = std::max(1.0 * score / (width * height), 0.1);
      }
    }
    // Check if snake collides with the wall
    if (snake[0].x < 0 || snake[0].x >= width || snake[0].y < 0 || snake[0].y >= height) {
      done = true;
    }
    // Check if snake collides with itself
    for (size_t i = 1; i < snake.size(); ++i) {
      if (snake[i].x == snake[0].x && snake[i].y == snake[0].y) {
        done = true;
      }
    }
  }

  void generateFood() {
    std::vector<std::pair<int, int>> avail_pos;
    // for (auto& pos : snake) {
    //   printf("(%d,%d) ", pos.x, pos.y);
    // } putchar('\n');
    for (int i = 0; i < width; ++i)
    for (int j = 0; j < height; ++j) {
      bool crash = false;
      for (auto& pos : snake) {
        if (pos.x == i && pos.y == j) { crash = true; break; }
      }
      if (!crash) avail_pos.emplace_back(i, j);
    }
    if (avail_pos.size() == 0) {  // win!
      done = true;
      food.x = food.y = -1;
    } else {
      std::uniform_int_distribution<int> dist(0, avail_pos.size()-1);
      auto& pos = avail_pos[dist(rng)];
      food.x = pos.first; food.y = pos.second;
    }
  }

  void render() {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Set background color to black
    SDL_RenderClear(renderer);

    // Draw snake
    SDL_Rect snakeRect;
    snakeRect.w = gridSize - 1;
    snakeRect.h = gridSize - 1;
    size_t size = snake.size();
    for (size_t i = 0; i < size; ++i) {
      int alpha = std::max(255.0 * (size - i) / size, 100.0);
      snakeRect.x = snake[i].x * gridSize;
      snakeRect.y = snake[i].y * gridSize;
      SDL_SetRenderDrawColor(renderer, 0, alpha, 0, 255);
      SDL_RenderFillRect(renderer, &snakeRect);
    }
    // for (const auto& segment : snake) {
    //   snakeRect.x = segment.x * gridSize;
    //   snakeRect.y = segment.y * gridSize;
    //   // printf("(%d %d) ", snakeRect.x, snakeRect.y);
    //   SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);  // Set color to green
    //   SDL_RenderFillRect(renderer, &snakeRect);
    // }

    // Draw food
    SDL_Rect foodRect;
    foodRect.w = gridSize - 1;
    foodRect.h = gridSize - 1;
    foodRect.x = food.x * gridSize;
    foodRect.y = food.y * gridSize;
    // printf("(%d %d) ", foodRect.x, foodRect.y);
    // putchar('\n');
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);  // Set color to red
    SDL_RenderFillRect(renderer, &foodRect);

    SDL_RenderPresent(renderer);
  }

  EnvInfo get_step_info() {
    std::vector<float> obs(obs_space);
    if (!done) {  // 如果done则返回全零的obs
      int n = snake.size();
      for (int i = 0; i < n; i++) {
        auto& pos = snake[i];
        obs[pos.x+pos.y*height] = -1.0f * i / n - 1;  // range=(-2, -1)
      }
      obs[food.x+food.y*height] = 1;
    }
    if (done) reward = reward_done;
    if (useRender) render();
    return EnvInfo(std::make_shared<std::vector<float>>(obs), reward, done);
  }
};
