#include "stochastic_decision_tree.h"

namespace kurumi {

StochasticDecisionTree::StochasticDecisionTree(const int depth, std::mt19937& randomEngine, const DTYPE etaWeights) :
  depth_(depth), etaWeights_(etaWeights), weights_(FEATURES * SPLIT_NODES), distributions_(1.0 / NUM_DECISIONS, LEAF_NODES * NUM_DECISIONS) {
      std::normal_distribution<DTYPE> normal(0, 0.01);
      std::generate(std::begin(weights_), std::end(weights_), [&]() { return normal(randomEngine); });
  }

void StochasticDecisionTree::loadModel(const std::string& filename) {
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ios::in | std::ios::binary);
    for (auto& x : weights_) {
        ifs.read(reinterpret_cast<char*>(&x), sizeof(DTYPE));
    }
    for (auto& x : distributions_) {
        ifs.read(reinterpret_cast<char*>(&x), sizeof(DTYPE));
    }
    ifs.close();
}

void StochasticDecisionTree::saveModel(const std::string& filename) {
    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ios::out | std::ios::binary);
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

int StochasticDecisionTree::leftNode(const int index) {
    return 2 * index + 1;
}

int StochasticDecisionTree::rightNode(const int index) {
    return 2 * index + 2;
}

void StochasticDecisionTree::updateWeights(const std::vector<std::pair<State, int> >& transitions) {
    std::valarray<DTYPE> dWeights(0.0, SPLIT_NODES * FEATURES);

    for (auto& transition : transitions) {
        const auto features = FeatureExtractor::extract(transition.first);
        const int y = transition.second;

        std::valarray<DTYPE> d(SPLIT_NODES);
        for (int i = 0; i < SPLIT_NODES; ++i) {
          d[i] = decisionFunction(features, i);
        }

        std::valarray<DTYPE> mu(NODES);
        mu[0] = 1;
        for (int i = 0; i < SPLIT_NODES; ++i) {
          mu[leftNode(i)] = d[i] * mu[i];
          mu[rightNode(i)] = (1 - d[i]) * mu[i];
        }

        const DTYPE yProb = (toValArray(mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)])
            * toValArray(distributions_[std::slice(y, LEAF_NODES, NUM_DECISIONS)])).sum();

        std::valarray<DTYPE> a(NODES);
        a[std::slice(SPLIT_NODES, LEAF_NODES, 1)] = toValArray(distributions_[std::slice(y, LEAF_NODES, NUM_DECISIONS)])
          * toValArray(mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)]) / (yProb + EPS);

        for (int i = SPLIT_NODES - 1; i >= 0; --i) {
          a[i] = a[leftNode(i)] + a[rightNode(i)];
          const auto dLpdfi = d[i] * a[rightNode(i)] - (1 - d[i]) * a[leftNode(i)];
          dWeights[std::slice(i * FEATURES, FEATURES, 1)] += dLpdfi * d[i] * (1 - d[i]) * features;
        }
    }

    weights_ -= etaWeights_ * dWeights;
}

void StochasticDecisionTree::updateDists(const std::vector<std::pair<State, int> >& transitions) {
    std::valarray<DTYPE> estimatedPi(1.0 / NUM_DECISIONS, NUM_DECISIONS * LEAF_NODES);

    std::vector<std::valarray<DTYPE> > mus;
    mus.reserve(transitions.size());
    for (auto& transition : transitions) {
        const auto features = FeatureExtractor::extract(transition.first);

        std::valarray<DTYPE> d(SPLIT_NODES);
        for (int i = 0; i < SPLIT_NODES; ++i) {
            d[i] = decisionFunction(features, i);
        }

        std::valarray<DTYPE> mu(NODES);
        mu[0] = 1;
        for (int i = 0; i < SPLIT_NODES; ++i) {
            mu[leftNode(i)] = d[i] * mu[i];
            mu[rightNode(i)] = (1 - d[i]) * mu[i];
        }

        mus.emplace_back(mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)]);
    }

    for (int l = 0; l < 20; ++l) {
        std::valarray<DTYPE> tempPi(0.0, NUM_DECISIONS * LEAF_NODES);
        for (unsigned int i = 0; i < transitions.size(); ++i) {
            const int y = transitions[i].second;
            const std::valarray<DTYPE> yProbs = mus[i] * toValArray(estimatedPi[std::slice(y, LEAF_NODES, NUM_DECISIONS)]);
            tempPi[std::slice(y, LEAF_NODES, NUM_DECISIONS)] += yProbs / (yProbs.sum() + EPS);
        }
        for (int i = 0; i < LEAF_NODES; ++i) {
            const auto sum = toValArray(tempPi[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)]).sum() + EPS;
            for (int y = 0; y < NUM_DECISIONS; ++y) {
                tempPi[i * NUM_DECISIONS + y] /= sum;
            }
        }
        tempPi += EPS;

        if (std::abs(tempPi - estimatedPi).sum() < 1e-3) {
            estimatedPi = tempPi;
            break;
        }
        estimatedPi = tempPi;
    }

    const auto logs = std::log(estimatedPi);
    std::valarray<DTYPE> target(NUM_DECISIONS * LEAF_NODES);
    for (int i = 0; i < LEAF_NODES; ++i) {
        target[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)] = logs[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)] - (logs[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)].sum() + 1) / NUM_DECISIONS;
    }

    //distributions_ += etaDists_ * (target - distributions_);
    distributions_ = target;
}

std::valarray<DTYPE> StochasticDecisionTree::toValArray(std::slice_array<DTYPE> ary) const {
    return static_cast<std::valarray<DTYPE> >(ary);
}

}
