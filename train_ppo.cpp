#include "env_snake.h"
#include "model/mlp.h"
#include "vec_env.h"
#include "argparse/argparse.hpp"
#include <tensorboard_logger/summary_writer.h>
#include <chrono>
#include <memory>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;
using tensorboard::get_current_timestamp;
using tensorboard::get_root_path;

struct Config {
  std::string version = "v3-retry";
  // std::string version = "debug";
  int cuda = false;
  int seed = 1;
  int torch_deterministic = true;
  double total_steps = 1e9;
  double init_lr = 2.5e-4;
  double final_lr = 2e-5;
  int game_size = 8;
  int num_envs = 64;
  int num_steps = 128;
  double anneal_steps = 8e8;
  double gamma = 0.995;
  double gae_lambda = 0.95;
  int num_minibatches = 8;
  int update_epochs = 3;
  int norm_adv = false;
  double clip_coef = 0.1;
  int clip_vloss = true;
  double init_ent_coef = 0.01;
  double final_ent_coef = 1e-5;
  double vf_coef = 0.5;
  double max_grad_norm = 0.5;
  int save_freq = int(1e7);
  std::string optimizer = "Adam";
  // Env
  double reward_step = -0.0;
  double reward_done = -0.0;
  double reward_food = NAN;  // dynamic

  int batch_size;
  int minibatch_size;
  int num_iterations;
  std::string run_name;
  std::string path_load_model;
  
  Config(int argc, char* argv[]) {
    argparse::ArgumentParser parser("ppo");
    parser.add_argument("--cuda").store_into(cuda).help("Set device to cuda, if cuda is availiable");
    parser.add_argument("--torch-deterministic").store_into(torch_deterministic).help("Make torch randomization results deterministic");
    parser.add_argument("--seed").store_into(seed).help("Random seed");
    parser.add_argument("--total-steps").store_into(total_steps).help("The total number of training steps");
    parser.add_argument("--init-lr").store_into(init_lr).help("The initialization of learning rate");
    parser.add_argument("--final-lr").store_into(final_lr).help("The final of learning rate");
    parser.add_argument("--game-size").store_into(game_size).help("The size of the game scene");
    parser.add_argument("--num-envs").store_into(num_envs).help("The number of threads in the parallel computing environment");
    parser.add_argument("--num-steps").store_into(num_steps).help("The number of steps for all environments in each training");
    parser.add_argument("--anneal-steps").store_into(anneal_steps).help("The number of steps for anneal learning rate and entropy coefficient");
    parser.add_argument("--gamma").store_into(gamma).help("The coefficient gamma in Markov process");
    parser.add_argument("--gae-lambda").store_into(gae_lambda).help("The coefficient lambda in GAE");
    parser.add_argument("--num-minibatches").store_into(num_minibatches).help("The number of minibatch for each model update");
    parser.add_argument("--update-epochs").store_into(update_epochs).help("The training epochs for each batch (all envs collect samples after 'num_steps')");
    parser.add_argument("--norm-adv").store_into(norm_adv).help("If triggered, use normalization for advantage value");
    parser.add_argument("--clip-coef").store_into(clip_coef).help("The clip coefficient of action scaling");
    parser.add_argument("--init-ent-coef").store_into(init_ent_coef).help("The coefficient of entropy loss");
    parser.add_argument("--final-ent-coef").store_into(final_ent_coef).help("The coefficient of entropy loss");
    parser.add_argument("--vf-coef").store_into(vf_coef).help("The coefficient of value loss");
    parser.add_argument("--max-grad-norm").store_into(max_grad_norm).help("The maximum gradient norm clip");
    parser.add_argument("--save-frequent").store_into(save_freq).help("The frequency of saving the model");
    parser.add_argument("--path-load-model").store_into(path_load_model).help("The path of loading the model");
    parser.add_argument("--reward-step").store_into(reward_step).help("The reward for each step");
    parser.add_argument("--reward-done").store_into(reward_done).help("The reward for episode done");
    parser.add_argument("--reward-food").store_into(reward_food).help("The reward for each food");
    parser.add_argument("--optimizer").store_into(optimizer).help("The name of optimizer: [Adam|SGD]");

    try {
      parser.parse_args(argc, argv);
    } catch(const std::exception& err) {
      std::cerr << err.what() << '\n';
      std::cerr << parser << '\n';
      std::exit(1);
    }

    batch_size = num_envs * num_steps;
    minibatch_size = batch_size / num_minibatches;
    run_name = version + "_seed" + std::to_string(seed) +
      "_size" + std::to_string(game_size) +
      "_" + get_current_timestamp();
  }
};

fs::path PATH_ROOT = get_root_path();

int main(int argc, char* argv[]) {
  Config cfg(argc, argv);
  srand(cfg.seed);
  torch::manual_seed(cfg.seed);
  at::globalContext().setDeterministicCuDNN(cfg.torch_deterministic ? true : false);
  // at::globalContext().setDeterministicAlgorithms(cfg.torch_deterministic ? true : false, true);
  
  fs::path path_tb_log = PATH_ROOT / "tb_logs" / cfg.run_name;
  tensorboard::SummaryWriter writer(path_tb_log);
  std::stringstream text;
  text << "|param|value|\n|-|-|\n";
  text << "|version|" << cfg.version << "|\n";
  text << "|seed|" << int(cfg.seed) << "|\n";
  text << "|total steps|" << int(cfg.total_steps) << "|\n";
  text << "|anneal steps|" << int(cfg.anneal_steps) << "|\n";
  text << "|init learning rate|" << cfg.init_lr << "|\n";
  text << "|final learning rate|" << cfg.final_lr << "|\n";
  text << "|game size|" << cfg.game_size << "|\n";
  text << "|init ent coef|" << cfg.init_ent_coef << "|\n";
  text << "|final ent coef|" << cfg.final_ent_coef << "|\n";
  text << "|load path|" << cfg.path_load_model << "|\n";
  text << "|reward step, done, food|" << cfg.reward_step << ", " << cfg.reward_done << ", " << cfg.reward_food << "|\n";
  text << "|minibatch size|" << cfg.minibatch_size << "|\n";
  text << "|num minibatchsizes size|" << cfg.num_minibatches << "|\n";
  text << "|clip vloss|" << cfg.clip_vloss << "|\n";
  text << "|optimizer|" << cfg.optimizer << "|\n";

  writer.add_text("hyperparameters", 0, text.str().c_str());
  fs::path path_ckpt = PATH_ROOT / "ckpt" / cfg.run_name;
  if (!fs::exists(path_ckpt)) fs::create_directories(path_ckpt);

  // Build environment
  VecEnv venv([cfg](int i){return std::make_shared<SnakeGame>(
    SnakeGameOption().seed(cfg.seed+i).width(cfg.game_size).height(cfg.game_size)
    .reward_step(cfg.reward_step).reward_done(cfg.reward_done).reward_food(cfg.reward_food)
  );}, cfg.num_envs);
  auto [obs_space, action_nums] = venv.get_space();
  // Build model, optimizer
  auto device(torch::cuda::is_available() && cfg.cuda ? torch::kCUDA : torch::kCPU);
  MLP model(obs_space, action_nums);
  model->to(device);
  std::shared_ptr<torch::optim::Optimizer> optimizer;
  if (cfg.optimizer == "Adam") {
    printf("Use Adam\n");
    optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(cfg.init_lr));
  } else if (cfg.optimizer == "SGD") {  // TODO: ERROR?
    printf("Use SGD\n");
    optimizer = std::make_shared<torch::optim::SGD>(model->parameters(), torch::optim::SGDOptions(cfg.init_lr));
  }
  // Load model
  int global_step = 0, pretrain_step = 0;
  if (cfg.path_load_model.size()) {
    pretrain_step = global_step = std::stoi(fs::path(cfg.path_load_model).stem());
    std::cout << "load from: " << cfg.path_load_model << '\n';
    torch::load(model, cfg.path_load_model);
  }
  cfg.num_iterations = (cfg.total_steps - pretrain_step) / cfg.batch_size;

  auto obs = torch::zeros({cfg.num_steps, cfg.num_envs, obs_space}).to(device);
  auto actions = torch::zeros({cfg.num_steps, cfg.num_envs, 1}).to(device);
  auto logprobs = torch::zeros({cfg.num_steps, cfg.num_envs}).to(device);
  auto rewards = torch::zeros({cfg.num_steps, cfg.num_envs}).to(device);
  auto dones = torch::zeros({cfg.num_steps, cfg.num_envs}).to(device);
  auto values = torch::zeros({cfg.num_steps, cfg.num_envs}).to(device);
  std::vector<double> total_reward(cfg.num_envs);
  std::vector<double> total_score(cfg.num_envs);
  std::vector<int> total_length(cfg.num_envs);

  auto start_time = std::chrono::high_resolution_clock::now();
  
  auto infos = venv.reset();
  std::vector<torch::Tensor> tensor_list;
  for (auto& info : infos) {
    tensor_list.push_back(torch::from_blob(info.obs->data(), obs_space));
  }
  auto next_obs = torch::stack(tensor_list, 0).to(device);
  auto next_done = torch::zeros(cfg.num_envs).to(device);

  for (int iteration = 1; iteration <= cfg.num_iterations + 1; ++iteration) {
    bool do_anneal = cfg.anneal_steps > 0 && global_step < cfg.anneal_steps;
    double frac = do_anneal ? (1.0 - 1.0 * global_step / cfg.anneal_steps) : 0.0;
    double lr = (cfg.init_lr - cfg.final_lr) * frac + cfg.final_lr;
    for (auto& group : optimizer->param_groups()) {
      if (group.has_options()) {
        static_cast<torch::optim::OptimizerOptions&>(group.options()).set_lr(lr);
      }
    }
    // static_cast<torch::optim::OptimizerOptions &>(optimizer->param_groups()[0].options()).set_lr(lr);
    double ent_coef = (cfg.init_ent_coef - cfg.final_ent_coef) * frac + cfg.final_ent_coef;
    double mean_reward = 0, mean_score = 0, mean_legnth = 0, done_count = 0;
    for (int step = 0; step < cfg.num_steps; ++step) {
      global_step += cfg.num_envs;
      obs[step] = next_obs;
      dones[step] = next_done;
      std::vector<int> action_vec(cfg.num_envs);
      {
        torch::NoGradGuard no_grad;
        auto [action, logprob, entropy, value] = model->get_action_and_value(next_obs);
        values[step] = value.view(-1);
        actions[step] = action.unsqueeze(-1);
        logprobs[step] = logprob;
        for (int i = 0; i < cfg.num_envs; i++)
          action_vec[i] = action[i].item<int>();
      }
      // (action.data<float>(), action.data<float>()+cfg.num_envs());
      // for (int i = 0; i < cfg.num_envs; ++i) printf("%d ", action_vec[i]);
      // printf("\n");
      infos = venv.step(action_vec);
      for (int i = 0; i < cfg.num_envs; ++i) {
        next_done[i] = (float)infos[i].done;
        next_obs[i] = torch::from_blob(infos[i].obs->data(), obs_space).to(device);
        rewards.index({step, i}) = (float)infos[i].reward;
        total_reward[i] += infos[i].reward;
        total_score[i] += double(infos[i].reward > 0);
        total_length[i] += 1;
        if (infos[i].done) {
          // printf("global_step=%d, total_reward=%.2lf, total_length=%d\n", global_step, total_reward[i], total_length[i]);
          // writer.add_scalar("charts/total_reward", global_step, total_reward[i]);
          // writer.add_scalar("charts/total_length", global_step, total_length[i]);
          done_count += 1;
          mean_reward += (total_reward[i] - mean_reward) / done_count;
          mean_score += (total_score[i] - mean_score) / done_count;
          mean_legnth += (total_length[i] - mean_legnth) / done_count;
          total_reward[i] = 0;
          total_score[i] = 0;
          total_length[i] = 0;
        }
      }
    }

    auto next_value = model->get_value(next_obs).view(-1);
    auto advantages = torch::zeros_like(rewards).to(device);
    {  // 计算GAE
      torch::NoGradGuard no_grad;
      Tensor nextnondone, nextvalues, delta;
      auto lastadvantages = torch::zeros(cfg.num_envs).to(device);
      for (int t = cfg.num_steps-1; t >= 0; --t) {
        if (t == cfg.num_steps - 1) {
          nextnondone = 1.0 - next_done;
          nextvalues = next_value;
        } else {
          nextnondone = 1.0 - dones[t+1];
          nextvalues = values[t+1];
        }
        delta = rewards[t] + cfg.gamma * nextvalues * nextnondone - values[t];
        advantages[t] = lastadvantages = delta + cfg.gamma * cfg.gae_lambda * lastadvantages;
      }
    }
    auto returns = advantages + values;

    auto b_obs = obs.view({-1, obs_space});
    auto b_logprobs = logprobs.view(-1);
    auto b_actions = actions.view(-1);
    auto b_advantages = advantages.view(-1);
    auto b_returns = returns.view(-1);
    auto b_values = values.view(-1);

    float mean_clipfracs = 0.0, approx_kl = 0.0;
    Tensor v_loss, pg_loss, entropy_loss;
    for (int epoch = 0; epoch < cfg.update_epochs; ++epoch) {
      auto b_idx = torch::randperm(b_obs.size(0)).to(device);
      for (int i = 0; i < cfg.num_minibatches; ++i) {
        auto mb_idx = b_idx.slice(0, i * cfg.minibatch_size, (i+1) * cfg.minibatch_size);
        auto [action, newlogprob, entropy, newvalue] = model->get_action_and_value(
          b_obs.index_select(0, mb_idx),
          b_actions.index_select(0, mb_idx).to(torch::kInt64)
        );
        auto logratio = newlogprob - b_logprobs.index_select(0, mb_idx);
        auto ratio = logratio.exp();
        {
          torch::NoGradGuard no_grad;
          int cnt = epoch * cfg.num_minibatches + i + 1;
          approx_kl += (
            ((ratio-1)-logratio).mean().item<float>() - approx_kl
          ) / cnt;
          mean_clipfracs += (
            ((ratio-1.0).abs() > cfg.clip_coef).to(torch::kFloat).mean().item<float>() - mean_clipfracs
          ) / cnt;
        }
        auto mb_advantages = b_advantages.index_select(0, mb_idx);
        if (cfg.norm_adv)
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-18);
        
        // Policy loss
        auto pg_loss1 = -mb_advantages * ratio;
        auto pg_loss2 = -mb_advantages * torch::clamp(ratio, 1-cfg.clip_coef, 1+cfg.clip_coef);
        pg_loss = torch::max(pg_loss1, pg_loss2).mean();

        // Value loss
        newvalue = newvalue.view(-1);
        auto v_loss_unclipped = (newvalue - b_returns.index_select(0, mb_idx)).pow(2);
        if (cfg.clip_vloss) {
          auto v_clipped = b_values.index_select(0, mb_idx) + torch::clamp(
            newvalue - b_values.index_select(0, mb_idx),
            -cfg.clip_coef, cfg.clip_coef
          );
          auto v_loss_clipped = (v_clipped - b_returns.index_select(0, mb_idx)).pow(2);
          v_loss = 0.5 * torch::max(v_loss_unclipped, v_loss_clipped).mean();
        } else {
          v_loss = 0.5 * v_loss_unclipped;
        }

        entropy_loss = entropy.mean();
        auto loss = pg_loss - ent_coef * entropy_loss + cfg.vf_coef * v_loss;

        optimizer->zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.max_grad_norm);
        optimizer->step();
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    float SPS = 1.0 * (global_step - pretrain_step) / duration.count() * 1e3;
    // printf("vloss=%.4f, ploss=%.4f, entropy=%.4f, approx_kl=%.4f, clipfrac=%.4f, SPS=%d, duration=%.4fs\n",
    //   v_loss.item<float>(), pg_loss.item<float>(), entropy_loss.item<float>(),
    //   approx_kl, mean_clipfracs, int(SPS),
    //   duration.count() / 1e3);
    printf("SPS=%d, time used=%.4fs/less=%.4fs\n", int(SPS), duration.count() / 1e3, 1.0f*(cfg.total_steps-global_step+pretrain_step)/SPS);
    writer.add_scalar("losses/value_loss", global_step, v_loss.item<float>());
    writer.add_scalar("losses/policy_loss", global_step, pg_loss.item<float>());
    writer.add_scalar("losses/entropy_loss", global_step, entropy_loss.item<float>());
    writer.add_scalar("losses/approx_kl", global_step, approx_kl);
    writer.add_scalar("losses/clipfracs", global_step, mean_clipfracs);
    writer.add_scalar("charts/learning_rate", global_step, lr);
    writer.add_scalar("charts/entropy_coefficient", global_step, ent_coef);
    writer.add_scalar("charts/SPS", global_step, SPS);
    writer.add_scalar("charts/total_reward", global_step, mean_reward);
    writer.add_scalar("charts/total_score", global_step, mean_score);
    writer.add_scalar("charts/total_length", global_step, mean_legnth);

    if (iteration == 1 || iteration == cfg.num_iterations + 1 ||
    ((global_step-pretrain_step) % cfg.save_freq < cfg.batch_size)) {
      printf("Save model: %d\n", global_step);
      std::string model_name = std::to_string(global_step);
      while (model_name.size() < 10) model_name = "0" + model_name;
      torch::save(model, path_ckpt / (model_name + ".pt"));
    }
  }
  writer.close();
  return 0;
}
