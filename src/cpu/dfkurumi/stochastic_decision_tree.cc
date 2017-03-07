#include "stochastic_decision_tree.h"

DEFINE_bool(greedy, false, "");

namespace kurumi {

StochasticDecisionTree::StochasticDecisionTree(const int depth, std::mt19937& randomEngine) :
  depth_(depth), randomEngine_(randomEngine), weights_(FEATURES * SPLIT_NODES), distributions_(1, LEAF_NODES * NUM_DECISIONS) {}

StochasticDecisionTree::~StochasticDecisionTree() {
    if (writer_) {
        shared_memory_object::remove(id_.c_str());
    }
}

void StochasticDecisionTree::setUpToLearn(const std::string& id, const DTYPE etaWeights, const DTYPE etaDists, const bool writer) {
    etaWeights_ = etaWeights;
    etaDists_ = etaDists;
    writer_ = writer;
    id_ = id;

    std::normal_distribution<DTYPE> normal(0, 0.01);
    std::generate(std::begin(weights_), std::end(weights_), [&]() { return normal(randomEngine_); });

    if (writer) {
        smo_ = shared_memory_object(open_or_create, id_.c_str(), read_write);
        smo_.truncate(MEMORY_SIZE);
        region_ = mapped_region(smo_, read_write);
    } else {
        smo_ = shared_memory_object(open_only, id_.c_str(), read_only);
        region_ = mapped_region(smo_, read_only);
    }
    sharedMemoryAddr_ = reinterpret_cast<DTYPE*>(region_.get_address());
    sync();
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

void StochasticDecisionTree::sync() {
    if (writer_) {
      writeToSharedMemory();
    } else {
      readFromSharedMemory();
    }
}

void StochasticDecisionTree::readFromSharedMemory() {
    std::copy(sharedMemoryAddr_,                   sharedMemoryAddr_ + weights_.size(),                         std::begin(weights_));
    std::copy(sharedMemoryAddr_ + weights_.size(), sharedMemoryAddr_ + weights_.size() + distributions_.size(), std::begin(distributions_));
}

void StochasticDecisionTree::writeToSharedMemory() {
    std::copy(std::begin(weights_),       std::end(weights_),       sharedMemoryAddr_);
    std::copy(std::begin(distributions_), std::end(distributions_), sharedMemoryAddr_ + weights_.size());
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

std::pair<Action, std::tuple<int, DTYPE, std::valarray<DTYPE> > > StochasticDecisionTree::selectAction(const State& state) const {
    const auto features = FeatureExtractor::extract(state);

    int n = 0;
    DTYPE mu = 1;
    while (n < SPLIT_NODES) {
        const DTYPE d = decisionFunction(features, n);
        //const bool branchToLeft = std::bernoulli_distribution(d)(randomEngine_);
        const bool branchToLeft = FLAGS_greedy ? d > 0.5 : std::bernoulli_distribution(d)(randomEngine_);
        mu *= branchToLeft ? d : (1 - d);
        n += n + 1 + (1 - static_cast<int>(branchToLeft));
    }

    const std::valarray<DTYPE> dist = pi(n - SPLIT_NODES) * state.legalActions;
    //const int id = std::discrete_distribution<int>(std::begin(dist), std::end(dist))(randomEngine_);
    const int id = FLAGS_greedy ? std::distance(std::begin(dist), std::max_element(std::begin(dist), std::end(dist))) : std::discrete_distribution<int>(std::begin(dist), std::end(dist))(randomEngine_);

    return {Action(id), std::make_tuple(n - SPLIT_NODES, mu, pi(n - SPLIT_NODES))};
}

int StochasticDecisionTree::leftNode(const int index) {
    return 2 * index + 1;
}

int StochasticDecisionTree::rightNode(const int index) {
    return 2 * index + 2;
}

void StochasticDecisionTree::update(const std::vector<std::pair<State, Action> >& transitions, const int reward) {
    std::valarray<DTYPE> dWeights(0.0, SPLIT_NODES * FEATURES);
    std::valarray<DTYPE> dDists(0.0, LEAF_NODES * NUM_DECISIONS);

    for (auto& transition : transitions) {
        const int y = transition.second.decisionId;
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

        /*std::valarray<DTYPE> probs(NUM_DECISIONS);
        for (int i = 0; i < NUM_DECISIONS; ++i) {
            probs[i] = (toValArray(mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)])
                       * toValArray(distributions_[std::slice(i, LEAF_NODES, NUM_DECISIONS)])).sum();
        }*/
        const DTYPE yProb = (toValArray(mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)])
                            * toValArray(distributions_[std::slice(y, LEAF_NODES, NUM_DECISIONS)])).sum();

        std::valarray<DTYPE> a(NODES);
        a[std::slice(SPLIT_NODES, LEAF_NODES, 1)] = toValArray(distributions_[std::slice(y, LEAF_NODES, NUM_DECISIONS)])
                                                    * toValArray(mu[std::slice(SPLIT_NODES, LEAF_NODES, 1)]) / (yProb + EPS);

        for (int i = 0; i < LEAF_NODES; ++i) {
            const std::valarray<DTYPE> dist = distributions_[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)];
            const auto e = std::exp(dist);
            const DTYPE sum = e.sum();
            const std::valarray<DTYPE> dDistsi = mu[i + SPLIT_NODES] * e[y] * e / (yProb + EPS);
            dDists[std::slice(NUM_DECISIONS * i, NUM_DECISIONS, 1)] -= dDistsi;
            dDists[NUM_DECISIONS * i + y] += dDistsi[y] + mu[i + SPLIT_NODES] * e[y] * (sum - e[y]) / (sum * sum * yProb + EPS);

            //a[i + SPLIT_NODES] -= lambda_ * (1 + (toValArray(distributions_[std::slice(i * NUM_DECISIONS, NUM_DECISIONS, 1)]) * std::log(probs)).sum());
        }

        for (int i = SPLIT_NODES - 1; i >= 0; --i) {
            a[i] = a[leftNode(i)] + a[rightNode(i)];
            const auto dLpdfi = d[i] * a[rightNode(i)] - (1 - d[i]) * a[leftNode(i)];
            dWeights[std::slice(i * FEATURES, FEATURES, 1)] += dLpdfi * d[i] * (1 - d[i]) * features;
        }
    };

    /*dWeights *= reward;
    dDists *= reward;

    const DTYPE etaPlus = 1.2;
    const DTYPE etaMinus = 0.5;
    const DTYPE deltaMax = 10;
    const DTYPE deltaMin = 0.00001;

    static const std::valarray<DTYPE> zeroWeights(0.0, weights_.size());
    etaWeights_[dWeights * prevdWeights_ > zeroWeights] = min(deltaMax, toValArray(etaPlus * toValArray(etaWeights_[dWeights * prevdWeights_ > zeroWeights])));
    etaWeights_[dWeights * prevdWeights_ < zeroWeights] = max(deltaMin, toValArray(etaMinus * toValArray(etaWeights_[dWeights * prevdWeights_ < zeroWeights])));
    deltaWeights_[dWeights * prevdWeights_ >= zeroWeights] = toValArray(sgn(dWeights)[dWeights * prevdWeights_ >= zeroWeights]) * toValArray(etaWeights_[dWeights * prevdWeights_ >= zeroWeights]);
    deltaWeights_[dWeights * prevdWeights_ < zeroWeights] = -toValArray(deltaWeights_[dWeights * prevdWeights_ < zeroWeights]);
    weights_ -= deltaWeights_;
    prevdWeights_ = dWeights;
    prevdWeights_[dWeights * prevdWeights_ < zeroWeights] = 0;

    static const std::valarray<DTYPE> zeroDists(0.0, distributions_.size());
    etaDists_[dDists * prevdDists_ > zeroDists] = min(deltaMax, toValArray(etaPlus * toValArray(etaDists_[dDists * prevdDists_ > zeroDists])));
    etaDists_[dDists * prevdDists_ < zeroDists] = max(deltaMin, toValArray(etaMinus * toValArray(etaDists_[dDists * prevdDists_ < zeroDists])));
    deltaDists_[dDists * prevdDists_ >= zeroDists] = toValArray(sgn(dDists)[dDists * prevdDists_ >= zeroDists]) * toValArray(etaDists_[dDists * prevdDists_ >= zeroDists]);
    deltaDists_[dDists * prevdDists_ < zeroDists] = -toValArray(deltaDists_[dDists * prevdDists_ < zeroDists]);
    distributions_ += deltaDists_;
    prevdDists_ = dDists;
    prevdDists_[dDists * prevdDists_ < zeroDists] = 0;

    weights_ -= etaWeights_ * dWeights * reward;
    distributions_ += etaDists_ * dDists * reward;*/

    const DTYPE LIMIT = 500.0;
    for (auto& x : distributions_) {
        x = std::min(LIMIT, std::max(-LIMIT, x)); // rectify to prevent overflow
    }
}

std::valarray<DTYPE> StochasticDecisionTree::toValArray(std::slice_array<DTYPE> ary) const {
    return static_cast<std::valarray<DTYPE> >(ary);
}

}
