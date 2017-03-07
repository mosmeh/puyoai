#ifndef KURUMI_PUPPET_AI_H_
#define KURUMI_PUPPET_AI_H_

#include <memory>
#include <string>

#include "core/client/ai/ai_base.h"
#include "core/client/ai/drop_decision.h"
#include "core/client/client_connector.h"
#include "core/frame_response.h"
#include "core/kumipuyo_seq.h"
#include "core/player_state.h"
#include "core/server/connector/server_connector.h"

class CoreField;
class PlainField;
struct FrameRequest;

// AI is a utility class of AI.
// You need to implement think() at least.
class PuppetAI : public AIBase {
public:
    virtual ~PuppetAI() override;
    const std::string& name() const { return name_; }

    void runLoop();

    // Set AI's behavior. If true, you can rethink next decision when the enemy has started his rensa.
    void setBehaviorRethinkAfterOpponentRensa(bool flag) { behaviorRethinkAfterOpponentRensa_ = flag; }

protected:
    PuppetAI(int argc, char* argv[], const std::string& name, const std::string& expertName);
    explicit PuppetAI(const std::string& name);

    // think will be called when AI should decide the next decision.
    // Basically, this will be called when NEXT2 has appeared.
    // |frameId| is the frameId that you will get to start moving your puyo.
    // In other words, it's a kind of 'frameInitiated'.
    // |fast| will be true when AI should decide the next decision immeidately,
    // e.g. ojamas are dropped or the enemy has fired some rensa
    // (if you set behavior). This might be also called when field is inconsistent
    // in wii_server.
    // If |fast| is true, you will have 30 ms to decide your hand.
    // Otherwise, you will have at least 300 ms to decide your hand.
    // |KumipuyoSeq| will have at least 2 kumipuyos. When we know more Kumipuyo sequence,
    // it might contain more. It's up to you if you will use >=3 kumipuyos.
    virtual DropDecision think(int frameId, const CoreField&, const KumipuyoSeq&,
                               const PlayerState& me, const PlayerState& enemy, bool fast) const = 0;

    // gaze will be called when AI should gaze the enemy's field.
    // |frameId| is the frameId where the enemy has started moving his puyo.
    // His moving puyo is the front puyo of the KumipuyoSeq.
    // KumipuyoSeq has at least 2 kumipuyos. When we know more Kumipuyo sequence,
    // it might contain more.
    // Since gaze might be called in the same frame as think(), you shouldn't consume
    // much time for gaze.
    virtual void gaze(int frameId, const CoreField& enemyField, const KumipuyoSeq&);

    // ----------------------------------------------------------------------
    // Callbacks. If you'd like to customize your AI, it is good if you could use
    // the following hook methods.
    // These callbacks will be called from the corresponding method.
    // i.e. onX() will be called from X().

    virtual void onGameWillBegin(const FrameRequest&) {}
    virtual void onGameHasEnded(const FrameRequest&) {}

    virtual void onPreDecisionRequestedForMe(const FrameRequest&) {}
    virtual void onDecisionRequestedForMe(const FrameRequest&) {}
    virtual void onGroundedForMe(const FrameRequest&) {}
    virtual void onPuyoErasedForMe(const FrameRequest&) {}
    virtual void onOjamaDroppedForMe(const FrameRequest&) {}
    virtual void onNext2AppearedForMe(const FrameRequest&) {}

    virtual void onDecisionRequestedForEnemy(const FrameRequest&) {}
    virtual void onGroundedForEnemy(const FrameRequest&) {}
    virtual void onPuyoErasedForEnemy(const FrameRequest&) {}
    virtual void onOjamaDroppedForEnemy(const FrameRequest&) {}
    virtual void onNext2AppearedForEnemy(const FrameRequest&) {}

    // Should rethink just before sending next decision.
    void requestRethink() { rethinkRequested_ = true; }

    // ----------------------------------------------------------------------
    // Usually, you don't need to care about methods below here.

    // |gameWillBegin| will be called just before a new game will begin.
    // FrameRequest might contain NEXT and NEXT2 puyos, but it's not guaranteed.
    // Please initialize your AI in this function.
    void gameWillBegin(const FrameRequest&);

    // |gameHasEnded| will be called just after a game has ended.
    void gameHasEnded(const FrameRequest&);

    void preDecisionRequestedForMe(const FrameRequest&);

    void decisionRequestedForMe(const FrameRequest&);
    void decisionRequestedForEnemy(const FrameRequest&);
    static void decisionRequestedForCommon(PlayerState* p1, PlayerState* p2);

    void groundedForMe(const FrameRequest&);
    void groundedForEnemy(const FrameRequest&);
    void groundedForCommon(PlayerState*, int frameId);

    void puyoErasedForMe(const FrameRequest&);
    void puyoErasedForEnemy(const FrameRequest&);
    static void puyoErasedForCommon(PlayerState* p1, PlayerState* p2, int frameId, const PlainField& provided);

    void ojamaDroppedForMe(const FrameRequest&);
    void ojamaDroppedForEnemy(const FrameRequest&);
    static void ojamaDroppedForCommon(PlayerState*);

    void next2AppearedForMe(const FrameRequest&);
    void next2AppearedForEnemy(const FrameRequest&);
    void next2AppearedForCommon(PlayerState*, const KumipuyoSeq&);

    const PlayerState& myPlayerState() const { return me_; }
    const PlayerState& enemyPlayerState() const { return enemy_; }

    PlayerState* mutableMyPlayerState() { return &me_; }
    PlayerState* mutableEnemyPlayerState() { return &enemy_; }

    FrameResponse masterResponse_;

private:
    friend class AITest;
    friend class Endless;
    friend class Solver;

    static bool isFieldInconsistent(const PlainField& ours, const PlainField& provided);
    static CoreField mergeField(const CoreField& ours, const PlainField& provided, bool ojamaDropped);

    // Returns the remembered sequence. If desynced, provided is returned as is.
    KumipuyoSeq rememberedSequence(int indexFrom, const KumipuyoSeq& provided) const;

    std::string name_;
    std::unique_ptr<ClientConnector> connector_;

    bool desynced_;

    bool rethinkRequested_;
    int enemyDecisionRequestFrameId_;

    PlayerState me_;
    PlayerState enemy_;

    bool behaviorRethinkAfterOpponentRensa_;

    std::unique_ptr<ServerConnector> masterConnector_;
};

#endif
