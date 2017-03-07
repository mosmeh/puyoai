#include <sstream>
#include <random>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "base/base.h"
#include "core/client/ai/ai.h"
#include "core/core_field.h"
#include "core/frame_request.h"

#include "stochastic_decision_tree.h"

namespace kurumi {

class DFKurumiAI : public AI {
public:
    DFKurumiAI(int argc, char* argv[], std::mt19937& randomEngine);

    DropDecision think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                       const PlayerState& me, const PlayerState& enemy, bool fast) const override;
    void gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq);

private:
    int games_;
    KumipuyoSeq enemySeq_;
    StochasticDecisionTree stochasticDecisionTree_;
    std::vector<std::pair<State, Action> > transitions_;

    void onGameHasEnded(const FrameRequest& req);
    int gameResultToInt(const GameResult result) const;
    void storeTransition(const std::pair<State, Action>& trans) const;
    void storeTransition(const std::pair<State, Action>& trans);
};

}
