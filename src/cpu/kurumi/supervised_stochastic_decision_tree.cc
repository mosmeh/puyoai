#include "supervised_stochastic_decision_tree.h"

namespace kurumi {

SupervisedStochasticDecisionTree::SupervisedStochasticDecisionTree(const int depth, const DTYPE lambda, std::mt19937& randomEngine) :
    StochasticDecisionTree(depth, randomEngine),
    lambda_(lambda),
    prevdWeights_(0.0, SPLIT_NODES * FEATURES),
    delta_(0.01, SPLIT_NODES * FEATURES),
    deltaW_(0.0, SPLIT_NODES * FEATURES) {}

int SupervisedStochasticDecisionTree::leftNode(const int index) {
    return 2 * index + 1;
}

int SupervisedStochasticDecisionTree::rightNode(const int index) {
    return 2 * index + 2;
}

void SupervisedStochasticDecisionTree::update(const std::vector<std::pair<State, int> >& transitions) {
    std::vector<std::valarray<DTYPE> > mus(transitions.size(), std::valarray<DTYPE>(LEAF_NODES)),
                                       ds(transitions.size(), std::valarray<DTYPE>(SPLIT_NODES));
#pragma omp parallel for
    for (unsigned int t = 0; t < transitions.size(); ++t) {
        const auto features = FeatureExtractor::extract(transitions[t].first);

        for (int i = 0; i < SPLIT_NODES; ++i) {
            ds[t][i] = decisionFunction(features, i);
        }

        std::valarray<DTYPE> mu(NODES);
        mu[0] = 1;
        for (int i = 0; i < SPLIT_NODES; ++i) {
            mu[leftNode(i)] = ds[t][i] * mu[i];
            mu[rightNode(i)] = (1 - ds[t][i]) * mu[i];
        }

        mus[t] = mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)];
    }

    std::valarray<DTYPE> estimatedPi(1.0 / NUM_DECISIONS, NUM_DECISIONS * LEAF_NODES);
    for (int l = 0; l < 20; ++l) {
        std::valarray<DTYPE> tempPi(0.0, NUM_DECISIONS * LEAF_NODES);
        for (unsigned int i = 0; i < transitions.size(); ++i) {
            const int y = transitions[i].second;
            const std::valarray<DTYPE> yProbs = mus[i] * toValArray(estimatedPi[std::slice(y, LEAF_NODES, NUM_DECISIONS)]);
            tempPi[std::slice(y, LEAF_NODES, NUM_DECISIONS)] += yProbs / (yProbs.sum() + EPS);
        }

        DTYPE diff = 0;
        for (int i = 0; i < LEAF_NODES; ++i) {
            std::valarray<DTYPE> dist = tempPi[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)];
            dist /= (dist.sum() + EPS);
            diff += std::abs(toValArray(estimatedPi[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)]) - dist).sum();
            estimatedPi[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)] = dist;
        }

        if (diff < 1e-5) {
            break;
        }
    }

    // inversion of softmax
    const auto logs = std::log(estimatedPi + EPS);
    for (int i = 0; i < LEAF_NODES; ++i) {
        const std::valarray<DTYPE> decisionLogs = logs[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)];
        distributions_[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)] = decisionLogs - (decisionLogs.sum() + 1) / NUM_DECISIONS;
    }

    std::valarray<DTYPE> dWeights(0.0, SPLIT_NODES * FEATURES);

#pragma omp parallel for
    for (unsigned int t = 0; t < transitions.size(); ++t) {
        const auto features = FeatureExtractor::extract(transitions[t].first);
        const int y = transitions[t].second;

        const std::valarray<DTYPE> yProbs = mus[t] * toValArray(estimatedPi[std::slice(y, LEAF_NODES, NUM_DECISIONS)]);

        std::valarray<DTYPE> a(NODES);
        a[std::slice(SPLIT_NODES, LEAF_NODES, 1)] = yProbs / (yProbs.sum() + EPS);

        for (int i = SPLIT_NODES - 1; i >= 0; --i) {
            a[i] = a[leftNode(i)] + a[rightNode(i)];
            const auto dLpdfi = ds[t][i] * a[rightNode(i)] - (1 - ds[t][i]) * a[leftNode(i)];
            dWeights[std::slice(i * FEATURES, FEATURES, 1)] += dLpdfi * features;
        }
    }

    dWeights /= transitions.size();

    // L1 regularization
    /*for (int i = 0; i < SPLIT_NODES; ++i) {
        for (int j = i * FEATURES + 1; j < (i + 1) * FEATURES; ++j) {
            int sign = 0;
            if (weights_[j] > 0) {
                sign = 1;
            } else if (weights_[j] < 0) {
                sign = -1;
            }
            dWeights[j] += lambda_ * sign;
        }
    }*/

    static const DTYPE etaPlus = 1.2, etaMinus = 0.5, deltaMax = 50, deltaMin = 1e-6;
    for (int i = 0; i < FEATURES * SPLIT_NODES; ++i) {
        if (prevdWeights_[i] * dWeights[i] > 0) {
            prevdWeights_[i] = dWeights[i];
            delta_[i] = std::min(delta_[i] * etaPlus, deltaMax);
            deltaW_[i] = (dWeights[i] > 0 ? 1 : -1) * delta_[i];
        } else if (prevdWeights_[i] * dWeights[i] < 0) {
            prevdWeights_[i] = 0;
            delta_[i] = std::max(delta_[i] * etaMinus, deltaMin);
            deltaW_[i] *= -1;
        } else {
            prevdWeights_[i] = dWeights[i];
            DTYPE sign = 0;
            if (dWeights[i] > 0) {
                sign = 1;
            } else if (dWeights[i] < 0) {
                sign = -1;
            }
            deltaW_[i] = sign * delta_[i];
        }
    }

    weights_ -= deltaW_;
}

std::valarray<DTYPE> SupervisedStochasticDecisionTree::toValArray(std::slice_array<DTYPE> ary) const {
    return static_cast<std::valarray<DTYPE> >(ary);
}

std::valarray<DTYPE> SupervisedStochasticDecisionTree::toValArray(std::mask_array<DTYPE> ary) const {
    return static_cast<std::valarray<DTYPE> >(ary);
}

}

