#ifndef CLIENT_CPU_MAYAH_MAYAH_AI_H_
#define CLIENT_CPU_MAYAH_MAYAH_AI_H_

#include <memory>
#include <string>
#include <vector>

#include "core/client/ai/ai.h"

#include "book_field.h"
#include "feature_parameter.h"
#include "gazer.h"

class CoreField;
class DropDecision;
class Evaluator;
class KumipuyoSeq;
class Plan;
class RefPlan;

class MayahAI : public AI {
public:
    static const int DEFAULT_DEPTH = 2;
    static const int DEFAULT_NUM_ITERATION = 3;
    static const int FAST_NUM_ITERATION = 1;

    MayahAI(int argc, char* argv[]);
    ~MayahAI();

    virtual void gameWillBegin(const FrameData&) override;
    virtual void gameHasEnded(const FrameData&) override;
    virtual DropDecision think(int frameId, const PlainField&, const KumipuyoSeq&) override;
    virtual DropDecision thinkFast(int frameId, const PlainField&, const KumipuyoSeq&) override;
    virtual void enemyGrounded(const FrameData&) override;
    virtual void enemyNext2Appeared(const FrameData&) override;

    // Use this directly in test. Otherwise, use via think/thinkFast.
    Plan thinkPlan(int frameId, const CoreField&, const KumipuyoSeq&, int depth, int maxIteration);

    void initializeGazerForTest(int frameId) { gazer_.initializeWith(frameId); }

protected:
    std::string makeMessageFrom(int frameId, const CoreField&, const KumipuyoSeq&, int maxIteration,
                                const Plan&, double thoughtTimeInSeconds) const;

    // For debugging purpose.
    void reloadParameter();

    std::unique_ptr<FeatureParameter> featureParameter_;
    std::vector<BookField> books_;
    Gazer gazer_;
    int thoughtMaxRensa_ = 0;
    int thoughtMaxScore_ = 0;
};

#endif
