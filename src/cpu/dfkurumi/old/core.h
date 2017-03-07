#ifndef DFKURUMI_CORE_H_
#define DFKURUMI_CORE_H_

#include <valarray>

#include "core/puyo_controller.h"
#include "core/field_constant.h"

namespace kurumi {

const int NUM_DECISIONS = 22;
const int FIELD_SIZE = FieldConstant::WIDTH * (FieldConstant::HEIGHT + 1);

const Decision DECISIONS[NUM_DECISIONS] = {
    Decision(2, 3), Decision(3, 3), Decision(3, 1), Decision(4, 1),
    Decision(5, 1), Decision(1, 2), Decision(2, 2), Decision(3, 2),
    Decision(4, 2), Decision(5, 2), Decision(6, 2),
    Decision(1, 1), Decision(2, 1), Decision(4, 3), Decision(5, 3),
    Decision(6, 3), Decision(1, 0), Decision(2, 0), Decision(3, 0),
    Decision(4, 0), Decision(5, 0), Decision(6, 0),
};

struct Action {
    Decision decision;
    int decisionId;

    Action(const int decisionId_) : decisionId(decisionId_) {
        decision = DECISIONS[decisionId];
    }

    Action(const Decision decision_) : decision(decision_) {
        decisionId = std::distance(DECISIONS, std::find(DECISIONS, DECISIONS + NUM_DECISIONS, decision));
    }
};

struct State {
    int frameId;
    PlayerState me;
    PlayerState enemy;
    std::valarray<float> legalActions;

    State(const int frameId_, const PlayerState& me_, const PlayerState& enemy_) : frameId(frameId_), me(me_), enemy(enemy_), legalActions(NUM_DECISIONS) {
        const int rep = me.seq.front().isRep() ? 11 : 22;
        for (int i = 0; i < rep; ++i) {
            legalActions[i] = static_cast<int>(PuyoController::isReachable(me.field, DECISIONS[i]));
        }
    }
};

}

#endif
