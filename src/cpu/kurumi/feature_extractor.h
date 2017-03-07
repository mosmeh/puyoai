#ifndef KURUMI_FEATURE_EXTRACTOR_H_
#define KURUMI_FEATURE_EXTRACTOR_H_

#include <array>
#include <valarray>

#include "core/plan/plan.h"
#include "core/field_constant.h"
#include "core/rensa/rensa_detector.h"

#include "core.h"

namespace kurumi {

const int PATTERN_FEATURES =  669;
const int FIELD_STATS = 29;
const int PLAN_FEATURES = 30 + FIELD_STATS;
const int FEATURES_FOR_ONE_PLAYER = 6 + PATTERN_FEATURES + FIELD_STATS + NUM_DECISIONS * PLAN_FEATURES * 3;
const int FEATURES = 4 + 2 * FEATURES_FOR_ONE_PLAYER;

class FeatureExtractor {
public:
    static std::valarray<DTYPE> extract(const State& state) {
        std::valarray<DTYPE> features(0.0, FEATURES);

        int index = 0;
        features[index++] = 1; // bias
        features[index++] = 1.0 / (state.frameId + 1);

        features[index++] = state.me.totalOjama(state.enemy) / ASSUMED_MAX_NUM_OJAMA;
        features[index++] = state.enemy.totalOjama(state.me) / ASSUMED_MAX_NUM_OJAMA;

        features[std::slice(index, FEATURES_FOR_ONE_PLAYER, 1)] = extractForOnePlayer(state.me, state.frameId);
        features[std::slice(index + FEATURES_FOR_ONE_PLAYER, FEATURES_FOR_ONE_PLAYER, 1)] = extractForOnePlayer(state.enemy, state.frameId);

        DCHECK(FEATURES == index + 2 * FEATURES_FOR_ONE_PLAYER);

        return features;
    }

private:
    static constexpr DTYPE FIELD_HEIGHT = FieldConstant::HEIGHT + 1;
    static constexpr DTYPE ASSUMED_MAX_FRAMES = 4000;
    static constexpr DTYPE ASSUMED_MAX_NUM_OJAMA = 4000;
    static constexpr DTYPE MAX_CHAINS = 19;

    static std::valarray<DTYPE> extractForOnePlayer(const PlayerState& pstate, const int frameId) {
        std::valarray<DTYPE> features(0.0, FEATURES_FOR_ONE_PLAYER);
        int index = 0;
        const int PATTERN_SIZE = 2;
        const int WIDTH = FieldConstant::WIDTH;
        for (int y = 0; y < FieldConstant::HEIGHT + 1; ++y) {
            for (int x = 0; x < FieldConstant::WIDTH; ++x) {
                for (int wx = std::max(0, x - PATTERN_SIZE); wx < std::min(WIDTH, x + PATTERN_SIZE + 1); ++wx) {
                    for (int wy = std::max(0, y - PATTERN_SIZE); wy < std::min(FieldConstant::HEIGHT + 1, y + PATTERN_SIZE + 1); ++wy) {
                        if (x + y * FieldConstant::WIDTH < wx + wy * FieldConstant::WIDTH) {
                            if (pstate.field.color(x, y) == pstate.field.color(wx, wy)) {
                                features[index] = 1;
                            }
                            index++;
                        }
                    }
                }
            }
        }

        features[index++] = static_cast<DTYPE>(pstate.isRensaOngoing());
        features[index++] = static_cast<DTYPE>(pstate.hasZenkeshi);

        features[index++] = (pstate.rensaFinishingFrameId() - frameId) / ASSUMED_MAX_FRAMES;

        features[index++] = pstate.noticedOjama() / ASSUMED_MAX_NUM_OJAMA;
        features[index++] = pstate.fixedOjama / ASSUMED_MAX_NUM_OJAMA;
        features[index++] = pstate.pendingOjama / ASSUMED_MAX_NUM_OJAMA;

        features[std::slice(index, FIELD_STATS, 1)] = extractFieldStats(pstate.field);
        index += FIELD_STATS;

        std::array<Plan, NUM_DECISIONS * 3> plans;
        std::array<bool, NUM_DECISIONS * 3> isAvailable = {false};

        Plan::iterateAvailablePlans(pstate.field, pstate.seq, 1, [&plans, &isAvailable](const RefPlan& plan) {
            int id = Action(plan.firstDecision()).decisionId;
            plans[id] = plan.toPlan();
            isAvailable[id] = true;
        });

        Plan::iterateAvailablePlans(pstate.field, pstate.seq, 2, [&plans, &isAvailable](const RefPlan& plan) {
            int id = Action(plan.firstDecision()).decisionId + NUM_DECISIONS;
            if (!isAvailable[id] || (isAvailable[id] && plan.totalOjama() > plans[id].totalOjama())) {
                plans[id] = plan.toPlan();
                isAvailable[id] = true;
            }

            id = Action(plan.firstDecision()).decisionId + 2 * NUM_DECISIONS;
            if (!isAvailable[id] || (isAvailable[id] && plan.totalFrames() < plans[id].totalFrames())) {
                plans[id] = plan.toPlan();
                isAvailable[id] = true;
            }
        });

        for (int i = 0; i < NUM_DECISIONS * 3; ++i) {
            if (isAvailable[i]) {
                const auto& plan = plans[i];
                features[index++] = plan.rensaResult().chains / MAX_CHAINS;
                for (int c = 0; c < 19; ++c) {
                    features[index++] = static_cast<DTYPE>(plan.rensaResult().chains > c);
                }
                features[index++] = static_cast<DTYPE>(plan.hasZenkeshi());
                features[index++] = plan.numChigiri() / 2.0;
                features[index++] = plan.framesToIgnite() / ASSUMED_MAX_FRAMES;
                features[index++] = plan.totalFrames() / ASSUMED_MAX_FRAMES;
                features[index++] = static_cast<DTYPE>(!plan.rensaResult().quick);
                features[index++] = plan.fallenOjama() / ASSUMED_MAX_NUM_OJAMA;
                features[index++] = plan.pendingOjama() / ASSUMED_MAX_NUM_OJAMA;
                features[index++] = plan.fixedOjama() / ASSUMED_MAX_NUM_OJAMA;
                features[index++] = plan.totalOjama() / ASSUMED_MAX_NUM_OJAMA;
                features[index++] = (plan.ojamaCommittingFrameId() - frameId) / ASSUMED_MAX_FRAMES;

                features[std::slice(index, FIELD_STATS, 1)] = extractFieldStats(plan.field());
                index += FIELD_STATS;
            } else {
                index += PLAN_FEATURES;
            }
        }

        DCHECK(index == FEATURES_FOR_ONE_PLAYER);

        return features;
    }

    static std::valarray<DTYPE> extractFieldStats(const CoreField& field) {
        std::valarray<DTYPE> stats(FIELD_STATS);
        int index = 0;
        stats[index++] = static_cast<DTYPE>(field.countReachableSpaces()) / FIELD_SIZE;
        stats[index++] = static_cast<DTYPE>(field.countUnreachableSpaces()) / FIELD_SIZE;
        stats[index++] = static_cast<DTYPE>(field.countPuyos()) / FIELD_SIZE;
        stats[index++] = static_cast<DTYPE>(field.countColor(PuyoColor::OJAMA)) / FIELD_SIZE;

        int count2, count3;
        field.countConnection(&count2, &count3);
        stats[index++] = static_cast<DTYPE>(count2) / FIELD_SIZE;
        stats[index++] = static_cast<DTYPE>(count3) / FIELD_SIZE;

        DTYPE sumHeight = 0;
        int unconnectedPuyos = 0;
        for (int x = 1; x <= FieldConstant::WIDTH; ++x) {
            stats[index++] = field.height(x) / FIELD_HEIGHT;
            stats[index++] = field.ridgeHeight(x) / FIELD_HEIGHT;
            stats[index++] = field.valleyDepth(x) / FIELD_HEIGHT;
            sumHeight += field.height(x);

            for (int y = 1; y <= FieldConstant::HEIGHT; ++y) {
                if (field.isNormalColor(x, y) && !field.isConnectedPuyo(x, y)) {
                    ++unconnectedPuyos;
                }
            }
        }
        stats[index++] = sumHeight / (FIELD_HEIGHT * FieldConstant::WIDTH);
        stats[index++] = static_cast<DTYPE>(unconnectedPuyos) / FIELD_SIZE;

        int max_chains = 0;
        int highest_ignition_y = 0;
        const bool prohibits[FieldConstant::MAP_WIDTH]{};
        RensaDetector::detectByDropStrategy(field, prohibits, PurposeForFindingRensa::FOR_FIRE, 2, 13,
                [&max_chains, &highest_ignition_y](CoreField&& complemented_field, const ColumnPuyoList& puyo_list)
        {
            int ignition_y = -1;
            for (int x = 1; x <= FieldConstant::WIDTH; ++x)
                if (puyo_list.sizeOn(x) > 0)
                    ignition_y = complemented_field.height(x);

            RensaResult rensa_result = complemented_field.simulate();
            if (std::make_tuple(rensa_result.chains, ignition_y) > std::make_tuple(max_chains, highest_ignition_y)) {
                max_chains = rensa_result.chains;
                highest_ignition_y = ignition_y;
            }
        });
        stats[index++] = max_chains / MAX_CHAINS;
        stats[index++] = max_chains / (MAX_CHAINS * field.countColorPuyos() + 1);
        stats[index++] = highest_ignition_y / FIELD_HEIGHT;

        return stats;
    }
};

}

#endif
