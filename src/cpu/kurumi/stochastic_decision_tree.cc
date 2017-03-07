#include "stochastic_decision_tree.h"

namespace kurumi {

StochasticDecisionTree::StochasticDecisionTree(const int depth, std::mt19937& randomEngine) :
  depth_(depth), randomEngine_(randomEngine), weights_(FEATURES * SPLIT_NODES), distributions_(1.0 / NUM_DECISIONS, LEAF_NODES * NUM_DECISIONS) {
      std::normal_distribution<DTYPE> normal(0, 0.01);
      std::generate(std::begin(weights_), std::end(weights_), [&]() { return normal(randomEngine); });
  }

void StochasticDecisionTree::loadModel(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    for (auto& x : weights_) {
        ifs.read(reinterpret_cast<char*>(&x), sizeof(DTYPE));
    }
    for (auto& x : distributions_) {
        ifs.read(reinterpret_cast<char*>(&x), sizeof(DTYPE));
    }
    ifs.close();
}

void StochasticDecisionTree::saveModel(const std::string& filename) {
    std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
    for (auto& x : weights_) {
        ofs.write(reinterpret_cast<char*>(&x), sizeof(DTYPE));
    }
    for (auto& x : distributions_) {
        ofs.write(reinterpret_cast<char*>(&x), sizeof(DTYPE));
    }
    ofs.close();
}

DTYPE StochasticDecisionTree::sigmoid(const DTYPE x) const {
    return 1 / (1 + std::exp(-x));
}

DTYPE StochasticDecisionTree::decisionFunction(const std::valarray<DTYPE>& features, const int n) const {
    const auto innerProduct = (features * weights_[std::slice(n * FEATURES, FEATURES, 1)]).sum();
    return sigmoid(innerProduct);
}

std::valarray<DTYPE> StochasticDecisionTree::pi(const int n) const {
    const std::valarray<DTYPE> dist = distributions_[std::slice(n * NUM_DECISIONS, NUM_DECISIONS, 1)];
    const auto e = std::exp(dist);
    return e / e.sum();
}

std::valarray<DTYPE> StochasticDecisionTree::predictDeterminately(const State& state) const {
    const auto features = FeatureExtractor::extract(state);

    std::valarray<DTYPE> d(SPLIT_NODES);
    for (int i = 0; i < SPLIT_NODES; ++i) {
        d[i] = decisionFunction(features, i);
    }

    std::valarray<DTYPE> mu(NODES);
    mu[0] = 1;
    for (int i = 0; i < SPLIT_NODES; ++i) {
        mu[2 * i + 1] = d[i] * mu[i];
        mu[2 * i + 2] = (1 - d[i]) * mu[i];
    }

    std::valarray<DTYPE> dist(0.0, NUM_DECISIONS);
    for (int i = 0; i < LEAF_NODES; ++i) {
        dist += pi(i) * mu[i + SPLIT_NODES];
    }

    return dist;
}

std::tuple<int, DTYPE, std::valarray<DTYPE> > StochasticDecisionTree::predict(const State& state) const {
    const auto features = FeatureExtractor::extract(state);

    int n = 0;
    DTYPE mu = 1;
    while (n < SPLIT_NODES) {
        const DTYPE d = decisionFunction(features, n);
        const bool branchToLeft = std::bernoulli_distribution(d)(randomEngine_);
        mu *= branchToLeft ? d : (1 - d);
        n += n + 1 + (1 - static_cast<int>(branchToLeft));
    }

    return std::make_tuple(n - SPLIT_NODES, mu, pi(n - SPLIT_NODES));
}

std::pair<Action, std::tuple<int, DTYPE, std::valarray<DTYPE> > > StochasticDecisionTree::selectAction(const State& state) const {
    int leafId;
    DTYPE mu;
    std::valarray<DTYPE> dist;
    std::tie(leafId, mu, dist) = predict(state);
    dist *= state.legalActions;
    const int id = std::discrete_distribution<int>(std::begin(dist), std::end(dist))(randomEngine_);

    return {Action(id), std::make_tuple(leafId, mu, dist)};
}

std::pair<Action, std::tuple<int, DTYPE, std::valarray<DTYPE> > > StochasticDecisionTree::selectActionGreedily(const State& state) const {
    int leafId;
    DTYPE mu;
    std::valarray<DTYPE> dist;
    std::tie(leafId, mu, dist) = predict(state);
    dist *= state.legalActions;
    const int id = std::distance(std::begin(dist), std::max_element(std::begin(dist), std::end(dist)));

    return {Action(id), std::make_tuple(leafId, mu, dist)};
}

}

