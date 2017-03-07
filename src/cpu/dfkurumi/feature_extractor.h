#ifndef DFKURUMI_FEATURE_EXTRACTOR_H_
#define DFKURUMI_FEATURE_EXTRACTOR_H_

#include <array>
#include <valarray>

#include "core/plan/plan.h"
#include "core/field_constant.h"
#include "core/rensa/rensa_detector.h"

#include "core.h"

namespace kurumi {

const int FIELD_COMBINATION_FEATURES = 3321; // combinations(FIELD_SIZE + 4, 2) = 3321
const int FIELD_FEATURES = FIELD_COMBINATION_FEATURES + 3 * FIELD_SIZE;
const int FEATURES_FOR_ONE_PLAYER = FIELD_FEATURES + 3 * 6 + 1 + 6 + 4 + 4;
const int PLAN_FEATURES = 12 + 6 * 3 + 1 + 6 + 3;
const int FEATURES = 4 + 2 * FEATURES_FOR_ONE_PLAYER + NUM_DECISIONS * PLAN_FEATURES;

class FeatureExtractor {
public:
    static std::valarray<DTYPE> extract(const State& state) {
        std::valarray<DTYPE> features(0.0, FEATURES);

        int index = 0;
        features[index++] = 1; // bias
        features[index++] = 1.0 / (state.frameId + 1);

        features[index++] = state.me.totalOjama(state.enemy) / ASSUMED_MAX_NUM_OJAMA;
        features[index++] = state.enemy.totalOjama(state.me) / ASSUMED_MAX_NUM_OJAMA;

        std::array<std::pair<bool, Plan>, NUM_DECISIONS> plans;
        std::fill(plans.begin(), plans.end(), std::make_pair(false, Plan()));

        Plan::iterateAvailablePlans(state.me.field, state.me.seq, 1, [&plans](const RefPlan& plan) {
            int id = Action(plan.firstDecision()).decisionId;
            if (!plans[id].first || (plans[id].first && plan.rensaResult().chains > plans[id].second.rensaResult().chains)) {
                plans[id] = std::make_pair(true, plan.toPlan());
            }
        });
        for (int i = 0; i < NUM_DECISIONS; ++i) {
            if (plans[i].first) {
                const auto& plan = plans[i].second;
                features[index++] = plan.rensaResult().chains / MAX_CHAINS;
                features[index++] = static_cast<DTYPE>(plan.hasZenkeshi());
                features[index++] = 1.0 / (plan.numChigiri() + 1);
                features[index++] = 1.0 / (plan.framesToIgnite() + 1);
                features[index++] = 1.0 / (plan.totalFrames() + 1);
                features[index++] = static_cast<DTYPE>(plan.field().countReachableSpaces()) / FIELD_SIZE;
                features[index++] = 1.0 / (plan.field().countUnreachableSpaces() + 1);
                features[index++] = plan.fallenOjama() / ASSUMED_MAX_NUM_OJAMA;
                features[index++] = plan.pendingOjama() / ASSUMED_MAX_NUM_OJAMA;
                features[index++] = plan.fixedOjama() / ASSUMED_MAX_NUM_OJAMA;
                features[index++] = plan.totalOjama() / ASSUMED_MAX_NUM_OJAMA;
                features[index++] = 1.0 / (plan.ojamaCommittingFrameId() - state.frameId + 1);

                DTYPE averageHeight = 0;
                for (int x = 1; x <= FieldConstant::WIDTH; ++x) {
                    features[index++] = plan.field().height(x) / FIELD_HEIGHT;
                    features[index++] = plan.field().ridgeHeight(x) / FIELD_HEIGHT;
                    features[index++] = plan.field().valleyDepth(x) / FIELD_HEIGHT;
                    averageHeight += plan.field().height(x);
                }
                averageHeight /= FieldConstant::WIDTH;

                features[index++] = averageHeight / FIELD_HEIGHT;
                for (int x = 1; x <= FieldConstant::WIDTH; ++x) {
                    features[index++] = (plan.field().height(x) - averageHeight) / FIELD_HEIGHT;
                }

                int max_chains = 0;
                int highest_ignition_y = 0;
                const bool prohibits[FieldConstant::MAP_WIDTH]{};
                RensaDetector::detectByDropStrategy(plan.field(), prohibits, PurposeForFindingRensa::FOR_FIRE, 2, 13,
                    [&max_chains, &highest_ignition_y](CoreField&& complemented_field, const ColumnPuyoList& puyo_list)
                {
                    int ignition_y = -1;
                    for (int x = 1; x <= FieldConstant::WIDTH; ++x)
                        if (puyo_list.sizeOn(x) > 0)
                            ignition_y = complemented_field.height(x);

                    RensaResult rensa_result = complemented_field.simulate();
                    if (std::make_tuple(rensa_result.chains, ignition_y) > std::make_tuple(max_chains, highest_ignition_y))
                    {
                        max_chains = rensa_result.chains;
                        highest_ignition_y = ignition_y;
                    }
                });
                features[index++] = max_chains / MAX_CHAINS;
                features[index++] = highest_ignition_y / FIELD_HEIGHT;
                features[index++] = (highest_ignition_y - averageHeight) / FIELD_HEIGHT;
            } else {
                index += PLAN_FEATURES;
            }
        }

        const auto meFeatures = extractForOnePlayer(state.me, state.frameId);
        features[std::slice(index, FEATURES_FOR_ONE_PLAYER, 1)] = meFeatures;

        const auto enemyFeatures = extractForOnePlayer(state.enemy, state.frameId);
        features[std::slice(index + FEATURES_FOR_ONE_PLAYER, FEATURES_FOR_ONE_PLAYER, 1)] = enemyFeatures;

        return features;
    }

private:
    static constexpr DTYPE FIELD_HEIGHT = static_cast<DTYPE>(FieldConstant::HEIGHT + 1);
    static constexpr DTYPE ASSUMED_MAX_FRAMES = 4000;
    static constexpr DTYPE ASSUMED_MAX_NUM_OJAMA = 4000;
    static constexpr DTYPE MAX_CHAINS = 19.0;

    static std::valarray<DTYPE> extractForOnePlayer(const PlayerState& pstate, const int frameId) {
        std::valarray<DTYPE> features(0.0, FEATURES);
        std::array<PuyoColor, FIELD_SIZE + 4> colors;
        for (int y = 0; y < FieldConstant::HEIGHT + 1; ++y) {
            for (int x = 0; x < FieldConstant::WIDTH; ++x) {
                colors[x + y * FieldConstant::WIDTH] = pstate.field.color(x, y);
            }
        }
        for (int i = 0; i < 2; ++i) {
            colors[FIELD_SIZE + 2 * i] = pstate.seq.get(i).axis;
            colors[FIELD_SIZE + 2 * i + 1] = pstate.seq.get(i).child;
        }

        int index = 0;
        for (int i = 0; i < FIELD_SIZE + 4; ++i) {
            for (int j = 0; j < i; ++j) {
                if (colors[i] == colors[j]) {
                  features[index] = 1;
                }
                ++index;
            }
        }

        for (int i = 0; i < FIELD_SIZE; ++i) {
            if (colors[i] == PuyoColor::EMPTY) {
                features[index] = 1;
            }
            if (colors[i] == PuyoColor::OJAMA) {
                features[index + FIELD_SIZE] = 1;
            }
            if (isNormalColor(colors[i])) {
                features[index + 2 * FIELD_SIZE] = 1;
            }
            ++index;
        }
        index += 2 * FIELD_SIZE;

        DTYPE averageHeight = 0;
        for (int x = 1; x <= FieldConstant::WIDTH; ++x) {
            features[index++] = pstate.field.height(x) / FIELD_HEIGHT;
            features[index++] = pstate.field.ridgeHeight(x) / FIELD_HEIGHT;
            features[index++] = pstate.field.valleyDepth(x) / FIELD_HEIGHT;
            averageHeight += pstate.field.height(x);
        }
        averageHeight /= FieldConstant::WIDTH;

        features[index++] = averageHeight / FIELD_HEIGHT;
        for (int x = 1; x <= FieldConstant::WIDTH; ++x) {
            features[index++] = (pstate.field.height(x) - averageHeight) / FIELD_HEIGHT;
        }

        features[index++] = static_cast<DTYPE>(pstate.field.countReachableSpaces()) / FIELD_SIZE;
        features[index++] = static_cast<DTYPE>(pstate.field.countUnreachableSpaces()) / FIELD_SIZE;
        features[index++] = static_cast<DTYPE>(pstate.isRensaOngoing());
        features[index++] = static_cast<DTYPE>(pstate.hasZenkeshi);

        features[index++] = (pstate.rensaFinishingFrameId() - frameId) / ASSUMED_MAX_FRAMES;

        features[index++] = pstate.noticedOjama() / ASSUMED_MAX_NUM_OJAMA;
        features[index++] = pstate.fixedOjama / ASSUMED_MAX_NUM_OJAMA;
        features[index++] = pstate.pendingOjama / ASSUMED_MAX_NUM_OJAMA;

        return features;
    }
};

}

#endif
