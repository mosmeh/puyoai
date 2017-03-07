#include "stochastic_decision_tree.h"

#include <iostream>
#include <iterator>

namespace kurumi {

StochasticDecisionTree::StochasticDecisionTree(int depth, float alpha, float beta, std::mt19937& randomEngine)
  : depth_(depth),
    alpha_(alpha),
    beta_(beta),
    randomEngine_(randomEngine),
    weights_(FEATURES * SPLIT_NODES),
    biases_(SPLIT_NODES),
    distributions_(1.0 / NUM_DECISIONS, LEAF_NODES * NUM_DECISIONS) {
    std::normal_distribution<float> normal(0, 0.01);
    std::generate(std::begin(weights_), std::end(weights_), [&]() { return normal(randomEngine_); });
    std::generate(std::begin(biases_), std::end(biases_), [&]() { return normal(randomEngine_); });
}

void StochasticDecisionTree::loadModel(const std::string& filename) {
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ios::in | std::ios::binary);
    for (auto& x : weights_) {
        ifs.read(reinterpret_cast<char*>(&x), 1);
    }
    for (auto& x : biases_) {
        ifs.read(reinterpret_cast<char*>(&x), 1);
    }
    for (auto& x : distributions_) {
        ifs.read(reinterpret_cast<char*>(&x), 1);
    }
    ifs.close();
}

void StochasticDecisionTree::saveModel(const std::string& filename) {
    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ios::out | std::ios::binary);
    for (auto& x : weights_) {
        ofs.write(reinterpret_cast<char*>(&x), 1);
    }
    for (auto& x : biases_) {
        ofs.write(reinterpret_cast<char*>(&x), 1);
    }
    for (auto& x : distributions_) {
        ofs.write(reinterpret_cast<char*>(&x), 1);
    }
    ofs.close();
}

float StochasticDecisionTree::sigmoid(const float x) const {
    return 1 / (1 + std::exp(-x));
}

float StochasticDecisionTree::decisionFunction(const std::valarray<float>& features, const int n) const {
    const auto innerProduct = (features * weights_[std::slice(n * FEATURES, FEATURES, 1)]).sum() + biases_[n];
    return sigmoid(innerProduct);
}

std::valarray<float> StochasticDecisionTree::pi(const int n) const {
    const std::valarray<float> dist = distributions_[std::slice(n * NUM_DECISIONS, NUM_DECISIONS, 1)];
    const auto e = std::exp(dist);
    return e / e.sum();
}

std::pair<Action, std::valarray<float> > StochasticDecisionTree::selectAction(const State& state) const {
    const auto features = FeatureExtractor::extract(state);

    int n = 0;
    while(n < SPLIT_NODES) {
        float d = decisionFunction(features, n);
        bool branchToRight = std::bernoulli_distribution(1 - d)(randomEngine_);
        n += n + 1 + static_cast<int>(branchToRight);
    }

    const std::valarray<float> dist = pi(n - SPLIT_NODES) * state.legalActions;
    const int id = std::discrete_distribution<int>(std::begin(dist), std::end(dist))(randomEngine_);

    return {Action(id), dist};
}

void StochasticDecisionTree::update(const std::vector<std::pair<State, Action> >& transitions, const int reward) {
    std::valarray<float> dweights(0.0, SPLIT_NODES * FEATURES);
    std::valarray<float> dbiases(0.0, SPLIT_NODES);
    std::valarray<float> ddist(0.0, LEAF_NODES * NUM_DECISIONS);

    for (auto& trans : transitions) {
        const int y = trans.second.decisionId;
        const auto features = FeatureExtractor::extract(trans.first);

        std::valarray<float> d(SPLIT_NODES);
        for (int i = 0; i < SPLIT_NODES; ++i) {
            d[i] = decisionFunction(features, i);
        }

        std::valarray<float> mu(NODES);
        mu[0] = 1;
        for (int i = 0; i < SPLIT_NODES; ++i) {
            mu[2 * i + 1] = d[i] * mu[i];
            mu[2 * i + 2] = (1 - d[i]) * mu[i];
        }

        const float yProb = (static_cast<std::valarray<float> >(mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)])
                            * static_cast<std::valarray<float> >(distributions_[std::slice(y, LEAF_NODES, NUM_DECISIONS)])).sum();

        for (int i = 0; i < LEAF_NODES; ++i) {
            const std::valarray<float> dist = distributions_[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)];
            const auto e = std::exp(dist);
            const float sum = e.sum();
            ddist[NUM_DECISIONS * i + y] += mu[i + SPLIT_NODES] * e[y] * (sum - e[y]) / (sum * sum * yProb + EPS);
        }

        std::valarray<float> a(NODES);
        a[std::slice(SPLIT_NODES, LEAF_NODES, 1)] = static_cast<std::valarray<float> >(distributions_[std::slice(y, LEAF_NODES, NUM_DECISIONS)])
                                                    * static_cast<std::valarray<float> >(mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)]) / (yProb + EPS);

        for (int i = SPLIT_NODES - 1; i >= 0; --i) {
            a[i] = a[2 * i + 1] + a[2 * i + 2];

            const auto dlpdfn = d[i] * a[2 * i + 2] - (1 - d[i]) * a[2 * i + 1];
            dbiases[i] += dlpdfn / (yProb + EPS);
            dweights[std::slice(i * FEATURES, FEATURES, 1)] += dlpdfn * features / (yProb + EPS);
        }
    }

    weights_ += alpha_ * dweights * reward;
    biases_ += alpha_ * dbiases * reward;
    distributions_ += beta_ * ddist * reward;

    /*for (int i = 0; i < LEAF_NODES; ++i) {
        auto dist = pi(i);
        std::cerr << i << ":";
        std::copy(std::begin(dist), std::end(dist), std::ostream_iterator<float>(std::cerr, ","));
        std::cerr << std::endl;
    }
    std::cerr << "----------------------------------------------" << std::endl;*/
}

}
