#pragma once

#include "torch/torch.h"
#include <tuple>

using namespace torch;

extern const int HIDDEN_DIM = 256;

class MLPImpl : public nn::Module {
 public:
  MLPImpl(int obs_space, int action_nums) : 
    policy_output(nn::LinearOptions(HIDDEN_DIM, action_nums).bias(false)),
    value_output(nn::LinearOptions(HIDDEN_DIM, 1).bias(false)) {
    feature_mlp->push_back(nn::Linear(obs_space, HIDDEN_DIM));
    feature_mlp->push_back(nn::ReLU());
    feature_mlp->push_back(nn::Linear(HIDDEN_DIM, HIDDEN_DIM));
    feature_mlp->push_back(nn::ReLU());
    register_module("feature_mlp", feature_mlp);
    register_module("policy_output", policy_output);
    register_module("value_output", value_output);
  }

  std::pair<Tensor, Tensor> forward(Tensor x) {
    Tensor z = feature_mlp->forward(x);
    Tensor logist = policy_output->forward(z);
    Tensor value = value_output->forward(z);
    return {logist, value};
  }

  std::tuple<Tensor, Tensor, Tensor, Tensor> get_action_and_value(Tensor x, Tensor action={}) {
    auto [logits, value] = forward(x);
    auto probs = softmax(logits, -1);
    if (action.numel() == 0) {
      action = multinomial(probs, 1).squeeze(1);
      // std::cout << "DEBUG: action.sizes()=" << action.sizes() << '\n';
    }
    auto logprob = log(probs.gather(-1, action.view({-1, 1}))).squeeze(1);
    // std::cout << "DEBUG: logprob.sizes()=" << logprob.sizes() << '\n';
    auto entropy = -(probs * log(probs + 1e-18)).sum(-1);
    // std::cout << action.sizes() << ' ' << logprob.sizes() << ' ' << entropy.sizes() << ' ' << value.sizes() << '\n';
    // sizes=[64] [64] [64] [64, 1]
    return std::make_tuple(action, logprob, entropy, value);
  }

  Tensor get_value(Tensor x) {
    Tensor z = feature_mlp->forward(x);
    Tensor value = value_output->forward(z);
    return value;
  }

 private:
  nn::Sequential feature_mlp;
  nn::Linear policy_output;
  nn::Linear value_output;
};

TORCH_MODULE(MLP);