#ifndef KURUMI_KURUMI_AI_H_
#define KURUMI_KURUMI_AI_H_

#include <random>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "base/base.h"
#include "core/client/ai/ai.h"
#include "core/core_field.h"
#include "core/frame_request.h"
#include "core/pattern/decision_book.h"

#include "stochastic_decision_tree.h"

namespace kurumi {

class KurumiAI : public AI {
public:
    KurumiAI(int argc, char* argv[]);

    DropDecision think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                       const PlayerState& me, const PlayerState& enemy, bool fast) const override;
    void gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq);

private:
    KumipuyoSeq enemySeq_;
    std::mt19937 randomEngine_;
    StochasticDecisionTree tree_;
    //MultiLayerPerceptron mlp_;
    DecisionBook decisionBook_;
};

}

#endif
