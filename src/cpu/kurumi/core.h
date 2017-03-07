#ifndef KURUMI_CORE_H_
#define KURUMI_CORE_H_

#include <valarray>

#include "base/base.h"
#include "core/puyo_controller.h"
#include "core/field_constant.h"
#include "core/client/ai/ai.h"
#include "core/plan/plan.h"

namespace kurumi {

//typedef double DTYPE;
typedef float DTYPE;

const int NUM_DECISIONS = 22;
const int FIELD_SIZE = FieldConstant::WIDTH * (FieldConstant::HEIGHT + 1);
const Decision DECISIONS[NUM_DECISIONS] = {
    Decision(1, 0), Decision(2, 0), Decision(3, 0), Decision(4, 0), Decision(5, 0), Decision(6, 0),
    Decision(1, 1), Decision(2, 1), Decision(3, 1), Decision(4, 1), Decision(5, 1),
    Decision(1, 2), Decision(2, 2), Decision(3, 2), Decision(4, 2), Decision(5, 2), Decision(6, 2),
                    Decision(2, 3), Decision(3, 3), Decision(4, 3), Decision(5, 3), Decision(6, 3)
};

struct Action {
    Decision decision;
    int decisionId;

    Action() {
        decision = Decision(3, 2);
        decisionId = 13;
    }

    Action(const int decisionId_) : decisionId(decisionId_) {
        decision = DECISIONS[decisionId];
    }

    Action(const Decision decision_) : decision(decision_) {
        decisionId = std::distance(DECISIONS, std::find(DECISIONS, DECISIONS + NUM_DECISIONS, decision));
    }

    Action simplify(const bool rep) const {
        if (rep && decisionId > 10) {
            return Action(decisionId - 11);
        } else {
            return *this;
        }
    }
};

struct State {
    int frameId;
    PlayerState me;
    PlayerState enemy;
    std::valarray<DTYPE> legalActions;

    State() {}
    State(const int frameId_, const PlayerState& me_, const PlayerState& enemy_) : frameId(frameId_), me(me_), enemy(enemy_), legalActions(NUM_DECISIONS) {
        std::fill(std::begin(legalActions), std::end(legalActions), 0);
        Plan::iterateAvailablePlans(me_.field, me_.seq, 1, [&](const RefPlan& plan) {
            if (plan.field().isEmpty(3, 12)) {
                legalActions[Action(plan.firstDecision()).simplify(me_.seq.front().isRep()).decisionId] = 1;
            }
        });

        /*for (int i = 0; i < NUM_DECISIONS; ++i) {
            legalActions[i] = static_cast<int>(PuyoController::isReachable(me.field, DECISIONS[i]));
        }*/
    }
};

}

#endif
