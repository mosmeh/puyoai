#include <sstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "base/base.h"
#include "core/plan/plan.h"
#include "core/client/ai/ai.h"
#include "core/core_field.h"
#include "core/frame_request.h"

#include "policy_network.h"

namespace kurumi {

class ReinforceKurumiAI : public AI {
public:
    ReinforceKurumiAI(int argc, char* argv[]);

    DropDecision think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                       const PlayerState& me, const PlayerState& enemy, bool fast) const override;
    void gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq);

private:
    int games;
    KumipuyoSeq enemySeq;
    PolicyNetwork policyNetwork_;
    std::vector<std::pair<State, Action> > transitions_;

    void onGameHasEnded(const FrameRequest& req);
    void storeTransition(const std::pair<State, Action>& trans) const;
    void storeTransition(const std::pair<State, Action>& trans);
};

}
