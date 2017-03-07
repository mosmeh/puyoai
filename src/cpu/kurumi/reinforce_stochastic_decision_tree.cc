#include "reinforce_stochastic_decision_tree.h"

namespace kurumi {

ReinforceStochasticDecisionTree::ReinforceStochasticDecisionTree(const int depth, std::mt19937& randomEngine, const std::string& id, const DTYPE etaWeights, const DTYPE etaDists, const DTYPE beta, const int minibatchSize, const bool writer) :
    StochasticDecisionTree(depth, randomEngine),
    etaWeights_(etaWeights),
    etaDists_(etaDists),
    beta_(beta),
    minibatchSize_(minibatchSize),
    id_(id),
    writer_(writer),
    mdWeights_(0.0, FEATURES * SPLIT_NODES),
    mdDists_(0.0, NUM_DECISIONS * LEAF_NODES) {

    if (writer) {
        smo_ = shared_memory_object(open_or_create, id.c_str(), read_write);
        smo_.truncate(MEMORY_SIZE * sizeof(DTYPE));
        region_ = mapped_region(smo_, read_write);
    } else {
        smo_ = shared_memory_object(open_only, id.c_str(), read_only);
        region_ = mapped_region(smo_, read_only);
    }
    sharedMemoryAddr_ = reinterpret_cast<DTYPE*>(region_.get_address());
}

ReinforceStochasticDecisionTree::~ReinforceStochasticDecisionTree() {
    if (writer_) {
        shared_memory_object::remove(id_.c_str());
    }
}

void ReinforceStochasticDecisionTree::readFromSharedMemory() {
    pool_.emplace_back(std::make_pair(weights_, distributions_));

    if (pool_.size() > 10) pool_.erase(pool_.begin());

    std::copy(sharedMemoryAddr_, sharedMemoryAddr_ + FEATURES * SPLIT_NODES, std::begin(weights_));
    std::copy(sharedMemoryAddr_ + FEATURES * SPLIT_NODES, sharedMemoryAddr_ + FEATURES * SPLIT_NODES + NUM_DECISIONS * LEAF_NODES, std::begin(distributions_));

    if (pool_.size() > 1) {
        unsigned int n = std::uniform_int_distribution<int>(0, pool_.size() - 2)(randomEngine_);
        std::tie(weights_, distributions_) = pool_[n];
    }
}

void ReinforceStochasticDecisionTree::writeToSharedMemory() {
    std::copy(std::begin(weights_),       std::end(weights_),       sharedMemoryAddr_);
    std::copy(std::begin(distributions_), std::end(distributions_), sharedMemoryAddr_ + FEATURES * SPLIT_NODES);
}

int ReinforceStochasticDecisionTree::leftNode(const int index) {
    return 2 * index + 1;
}

int ReinforceStochasticDecisionTree::rightNode(const int index) {
    return 2 * index + 2;
}

void ReinforceStochasticDecisionTree::update(const std::vector<std::pair<State, int> >& transitions, const DTYPE reward) {
    std::valarray<DTYPE> dWeights(0.0, SPLIT_NODES * FEATURES),
                         dWeightsH(0.0, SPLIT_NODES * FEATURES),
                         dDists(0.0, LEAF_NODES * NUM_DECISIONS);

#pragma omp parallel for
    for (unsigned int t = 0; t < transitions.size(); ++t) {
        const int y = transitions[t].second;
        const auto features = FeatureExtractor::extract(transitions[t].first);

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

        DTYPE yProb = 0;
        for (int i = 0; i < LEAF_NODES; ++i) {
            yProb += mu[SPLIT_NODES + i] * pi(i)[y];
        }

        std::valarray<DTYPE> a(NODES), b(NODES);

        for (int i = 0; i < LEAF_NODES; ++i) {
            const auto dist = pi(i);

            dDists[std::slice(NUM_DECISIONS * i, NUM_DECISIONS, 1)] -= mu[i + SPLIT_NODES] * dist[y] * dist / (yProb + EPS);
            dDists[NUM_DECISIONS * i + y] += mu[i + SPLIT_NODES] * dist[y] / (yProb + EPS);

            a[SPLIT_NODES + i] = dist[y] * mu[SPLIT_NODES + i] / (yProb + EPS);
            b[SPLIT_NODES + i] = mu[SPLIT_NODES + i] * (1 + std::log(mu[SPLIT_NODES + i] + EPS));
        }

        for (int i = SPLIT_NODES - 1; i >= 0; --i) {
            a[i] = a[leftNode(i)] + a[rightNode(i)];
            b[i] = b[leftNode(i)] + b[rightNode(i)];

            const auto dLpdfiA = d[i] * a[rightNode(i)] - (1 - d[i]) * a[leftNode(i)];
            const auto dLpdfiB = d[i] * b[rightNode(i)] - (1 - d[i]) * b[leftNode(i)];
            dWeights[std::slice(i * FEATURES, FEATURES, 1)] += dLpdfiA * features;
            dWeightsH[std::slice(i * FEATURES, FEATURES, 1)] -= dLpdfiB * features;
        }
    }

    mdWeights_ += dWeights / static_cast<DTYPE>(transitions.size()) * reward + beta_ * dWeightsH / transitions.size();
    mdDists_ += dDists / static_cast<DTYPE>(transitions.size()) * reward;
    count_++;

    if (count_ % minibatchSize_ == 0) {
        weights_ -= etaWeights_ * mdWeights_ / minibatchSize_;
        distributions_ += etaDists_ * mdDists_ / minibatchSize_;

        std::fill(std::begin(mdWeights_), std::end(mdWeights_), 0.0);
        std::fill(std::begin(mdDists_), std::end(mdDists_), 0.0);

        count_ = 0;
    }
}

std::valarray<DTYPE> ReinforceStochasticDecisionTree::toValArray(std::slice_array<DTYPE> ary) const {
    return static_cast<std::valarray<DTYPE> >(ary);
}

}

