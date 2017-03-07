#include <gflags/gflags.h>
#include <glog/logging.h>

#include "base/base.h"
#include "core/plan/plan.h"
#include "core/client/ai/ai.h"
#include "core/core_field.h"
#include "core/frame_request.h"
#include "core/rensa/rensa_detector.h"
#include "core/puyo_controller.h"

#include <random>
#include <iostream>

DEFINE_int32(seed, 1, "");

class RandomAI : public AI {
public:
    RandomAI(int argc, char* argv[]) : AI(argc, argv, "random"), randomEngine(FLAGS_seed) {}
    ~RandomAI() override {}

    DropDecision think(int frameId, const CoreField& f, const KumipuyoSeq& seq,
                       const PlayerState& me, const PlayerState& enemy, bool fast) const override
    {
        UNUSED_VARIABLE(frameId);
        UNUSED_VARIABLE(f);
        UNUSED_VARIABLE(me);
        UNUSED_VARIABLE(enemy);
        UNUSED_VARIABLE(fast);
        UNUSED_VARIABLE(seq);

        std::vector<Decision> legal;
        Plan::iterateAvailablePlans(f, seq, 1, [&](const RefPlan& plan) {
            if (plan.field().isEmpty(3, 12)) {
                legal.push_back(plan.firstDecision());
            }
        });

        if (legal.empty()) {
          return DropDecision(Decision(3, 2));
        } else {
        return DropDecision(legal.at(r(legal.size())));
        }
    }

private:
    std::mt19937 randomEngine;

    int r(int n) const {
        return const_cast<RandomAI*>(this)->r(n);
    }

    int r(int n) {
        return std::uniform_int_distribution<int>(0, n - 1)(randomEngine);
    }
};

int main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
#if !defined(_MSC_VER)
    google::InstallFailureSignalHandler();
#endif

    RandomAI(argc, argv).runLoop();
    return 0;
}
