#ifndef KURUMI_LOGISTIC_REGRESSOR_
#define KURUMI_LOGISTIC_REGRESSOR_

#include <valarray>

#include "core.h"
#include "feature_extractor.h"

namespace kurumi {

class LogisticRegressor {
public:
    LogisticRegressor(DTYPE eta, std::mt19937& randomEngine) : eta_(eta),
    weights_(NUM_DECISIONS * FEATURES),
    prevWeights_(NUM_DECISIONS * FEATURES),
    deltaW_(NUM_DECISIONS * FEATURES),
    delta_(NUM_DECISIONS * FEATURES), randomEngine_(randomEngine) {
        std::normal_distribution<DTYPE> normal(0, 0.01);
        std::generate(std::begin(weights_), std::end(weights_), [&]{ return normal(randomEngine_); });
    }

    std::valarray<DTYPE> predict(const State& state) const {
        const auto features = FeatureExtractor::extract(state);
        std::valarray<DTYPE> values(NUM_DECISIONS);
        for (int i = 0; i < NUM_DECISIONS; ++i) {
            values[i] = (static_cast<std::valarray<DTYPE> >(weights_[std::slice(i * FEATURES, FEATURES, 1)]) * features).sum();
        }
        std::valarray<DTYPE> probs = std::exp(values) * state.legalActions;
        probs /= probs.sum();
        return probs;
    }

    Action selectAction(const State& state) const {
        const std::valarray<DTYPE> dist = predict(state) * state.legalActions;
        return Action(std::discrete_distribution<int>(std::begin(dist), std::end(dist))(randomEngine_));
    }

    void update(const std::vector<std::pair<State, int> >& transitions, const DTYPE reward) {
        std::valarray<DTYPE> dWeights(0.0, NUM_DECISIONS * FEATURES);
#pragma omp parallel for
        for (unsigned int t = 0; t < transitions.size(); ++t) {
            const auto features = FeatureExtractor::extract(transitions[t].first);
            const int y = transitions[t].second;

            std::valarray<DTYPE> values(NUM_DECISIONS);
            for (int i = 0; i < NUM_DECISIONS; ++i) {
                values[i] = (static_cast<std::valarray<DTYPE> >(weights_[std::slice(i * FEATURES, FEATURES, 1)]) * features).sum();
            }
            std::valarray<DTYPE> dist = std::exp(values);
            dist /= dist.sum();

            dWeights[std::slice(y * FEATURES, FEATURES, 1)] += features;
            for (int i = 0; i < NUM_DECISIONS; ++i) {
                dWeights[std::slice(i * FEATURES, FEATURES, 1)] -= features * dist[i];
            }
        }

        dWeights *= reward;
        dWeights /= transitions.size();

        static const DTYPE etaPlus = 1.2, etaMinus = 0.5, deltaMax = 50, deltaMin = 1e-6;
#pragma omp parallel for
        for (int i = 0; i < FEATURES * NUM_DECISIONS; ++i) {
            dWeights[i] /= minibatchSize;
            if (prevWeights_[i] * dWeights[i] > 0) {
                prevWeights_[i] = dWeights[i];
                delta_[i] = std::min(delta_[i] * etaPlus, deltaMax);
                deltaW_[i] = (dWeights[i] > 0 ? 1 : -1) * delta_[i];
            } else if (prevWeights_[i] * dWeights[i] < 0) {
                prevWeights_[i] = 0;
                delta_[i] = std::max(delta_[i] * etaMinus, deltaMin);
                deltaW_[i] *= -1;
            } else {
                prevWeights_[i] = dWeights[i];
                DTYPE sign = 0;
                if (dWeights[i] > 0) {
                    sign = 1;
                } else if (dWeights[i] < 0) {
                    sign = -1;
                }
                deltaW_[i] = sign * delta_[i];
            }
        }

        weights_ += deltaW_;
    }

private:
    const DTYPE eta_;
    std::valarray<DTYPE> weights_, prevWeights_, deltaW_, delta_;
    int count_ = 0;
    std::mt19937& randomEngine_;
};

}

#endif
