#ifndef KURUMI_DUEL_RECORDER_AI_H_
#define KURUMI_DUEL_RECORDER_AI_H_

#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "base/base.h"
#include "core/client/ai/ai.h"
#include "core/core_field.h"

#include "puppet_ai.h"
#include "core.h"

namespace kurumi {

class DuelRecorderAI : public PuppetAI {
public:
    DuelRecorderAI(int argc, char* argv[]);
    ~DuelRecorderAI();

    DropDecision think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                       const PlayerState& me, const PlayerState& enemy, bool fast) const override;
    void gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq& seq);

private:
    int games_;
    KumipuyoSeq enemySeq_;
    std::ofstream ofs_;

    void saveTransition(const State& state) const;
    void saveTransition(const State& state);
};

}

#endif
