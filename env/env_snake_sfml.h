#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <memory>
#include <cassert>
#include "env.h"

const int width = 8;  // 游戏宽度格子
const int height = 8;  // 游戏高度格子
extern const int GAME_SIZE = width;
const int gridSize = 40;
const int windowWidth = width * gridSize;
const int windowHeight = height * gridSize;

enum Direction { Up, Down, Left, Right };

struct SnakeSegment {
  int x, y;
  SnakeSegment(int x, int y):x(x), y(y) {}
};

class SnakeGame: public Env{
 public:
  SnakeGame(bool useRender=false):
      Env(width * height, 4), useRender(useRender), score(0), deadCount(0) {
    srand(static_cast<unsigned>(time(0)));
    if (useRender) {
      window = std::make_unique<sf::RenderWindow>(sf::VideoMode(windowWidth, windowHeight), "Snake");
      foodEaten = false;
      // if (!font.loadFromFile("../Times_New_Roman.ttf")) {
      //   std::cerr << "Error loading font\n";
      // }
      // scoreText.setFont(font);
      // scoreText.setCharacterSize(u_int(gridSize/20*40));
      // scoreText.setFillColor(sf::Color::White);
      // scoreText.setPosition(0, 0);
    }
    generateFood();
    reset();
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
    score = 0;
    reward = -0.01;
    done = false;
    snake.clear();
    snake.emplace_back(width / 2, height / 2);
    direction = static_cast<Direction>(rand() % (Right+1));
    if (useRender) render();
    return get_step_info();
  }

  void play() {
    if (!useRender) {
      printf("Render MODE hasn't been started! Can't play...Exit\n");
      exit(0);
    }
    while (window->isOpen()) {
      handleInput(true);
      update();
      if (done) reset();
      else render();
      sf::sleep(sf::milliseconds(100));
    }
  }

 private:
  std::unique_ptr<sf::RenderWindow> window;
  std::vector<SnakeSegment> snake;
  Direction direction;
  sf::Vector2i food;
  bool foodEaten, done, useRender;
  int score, deadCount;
  double reward;
  // sf::Font font;
  // sf::Text scoreText;

  void handleInput(bool humanPlay=false) {
    sf::Event event;
    while (window->pollEvent(event)) {
      if (event.type == sf::Event::Closed) window->close();
      if (humanPlay) {
        if (event.type == sf::Event::KeyPressed) {
          switch (event.key.code) {
            case sf::Keyboard::Up: if (direction != Down) direction = Up; break;
            case sf::Keyboard::Down: if (direction != Up) direction = Down; break;
            case sf::Keyboard::Left: if (direction != Right) direction = Left; break;
            case sf::Keyboard::Right: if (direction != Left) direction = Right; break;
            default: break;
          }
        }
      }
    }
  }

  void update() {
    reward = -0.01;
    moveSnake();
    checkCollisions();
    if (foodEaten) {
      score += 1;
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
      reward = 3.0;
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
    bool crash;
    do {
      food.x = rand() % width;
      food.y = rand() % height;
      crash = false;
      for (auto& pos : snake)
        if (pos.x == food.x && pos.y == food.y) crash = true;
    }while(crash);
  }

  void render() {
    window->clear();

    // Draw score text
    // scoreText.setString(std::to_string(score));
    // window->draw(scoreText);

    // Draw snake
    sf::RectangleShape snakeRect(sf::Vector2f(gridSize - 1, gridSize - 1));
    snakeRect.setFillColor(sf::Color::Green);
    for (const auto& segment : snake) {
      snakeRect.setPosition(segment.x * gridSize, segment.y * gridSize);
      window->draw(snakeRect);
    }

    // Draw food
    sf::RectangleShape foodRect(sf::Vector2f(gridSize - 1, gridSize - 1));
    foodRect.setFillColor(sf::Color::Red);
    foodRect.setPosition(food.x * gridSize, food.y * gridSize);
    window->draw(foodRect);

    window->display();
  }

  EnvInfo get_step_info() {
    std::vector<float> obs(obs_space);
    if (!done) {  // 如果done则返回全零的obs
      for (auto& pos : snake)
        obs[pos.x+pos.y*height] = -1;
      obs[food.x+food.y*height] = 1;
    }
    if (done) reward = -1;
    if (useRender) render();
    #if 0  // DEBUG: 直接打印当前状态
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        char c = '#';
        for (auto &pos : snake) if (pos.x == j and pos.y == i) {c = 'x'; break;}
        if (food.x == j and food.y == i) c = '*';
        putchar(c);
      } putchar('\n');
    }
    putchar('\n');
    #endif
    return EnvInfo(std::make_shared<std::vector<float>>(obs), reward, done);
  }
};
