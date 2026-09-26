// UAIManager.h
#pragma once
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MCTSNode.h"
#include "UAI_InferenceQueue.h"
#include "UAI_RDGCache.h"
#include "AIInferenceSpec.h"
#include "NNEModelData.h"
#include "UAI_ReplayBufferManager.h"
#include "UAI_CombatSpec.h"

#include "NNE.h"
#include "NNERuntimeRDG.h"
#include "NNERuntime.h"
#include "UAIManager.generated.h"

// ---- HARD RULE ----
// PolicyPrior is NEVER modified by chance resolution.
// Only STATE transitions are affected by stochastic combat.
// NN outputs remain the sole source of policy distribution.

USTRUCT(BlueprintType)
struct FTrainingStats
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AI|Training")
    int32 SamplesInMemory = 0;

    UPROPERTY(BlueprintReadOnly, Category = "AI|Training")
    int32 SamplesOnDisk = 0;

    UPROPERTY(BlueprintReadOnly, Category = "AI|Training")
    int32 CurrentEpisodeId = 0;

    UPROPERTY(BlueprintReadOnly, Category = "AI|Training")
    int32 CurrentGameSampleCount = 0;

    UPROPERTY(BlueprintReadOnly, Category = "AI|Training")
    int32 CurrentStep = 0;

    UPROPERTY(BlueprintReadOnly, Category = "AI|Training")
    int32 TotalSteps = 0;
};

UENUM()
enum class EMCTSMode : uint8
{
    Training,
    Evaluation
};

USTRUCT()
struct FMCTSContext
{
    GENERATED_BODY()

    EMCTSMode Mode = EMCTSMode::Training;
    int32     RootPlayerId = -1;
    bool      bIsChanceNode = false;
    bool      bRootInitialized = false;
    float     DirichletEpsilon = 0.25f;
    float     DirichletAlpha = 0.3f;
    int32     SimulationCount = 400;
};

struct FInferenceBatchToken
{
    int32 BatchIndex;
    int32 NodeIndex;
    int32 PhaseId;
    int32 PlayerId;
};

// ------------------------------------------------------------------
// FApplyActionResult
// Returned by Blueprint-implemented events SimulateTransition,
// GetLegalActionMask, and GetCasualtyAssignmentMask.
// OutEntityLists replaces OutMovementContext — Blueprint writes the
// complete updated entity list for all territories into this array
// after applying an action. C++ stores it on the child node.
// OutEntityLists is parallel to the territory array: index T holds
// the updated FTerritoryEntityList for territory T.
// Blueprint need only populate OutEntityLists; C++ ignores it for
// chain-intermediate phases where SimulateTransition is not called.
// ------------------------------------------------------------------
USTRUCT(BlueprintType)
struct FApplyActionResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite)
    TArray<float> OutState;

    UPROPERTY(BlueprintReadWrite)
    int32 OutPhaseId = 0;

    UPROPERTY(BlueprintReadWrite)
    int32 OutPlayerId = 0;

    // Blueprint-owned legality mask — sole source of truth for legal actions.
    UPROPERTY(BlueprintReadWrite)
    TArray<bool> OutLegalActionMask;

    UPROPERTY(BlueprintReadWrite)
    bool bIsTerminal = false;

    // Updated entity lists after applying the action.
    // Blueprint populates this in SimulateTransition for all phases.
    // C++ stores it on the child node.
    // Indexed by territory: OutEntityLists[T] = entities at territory T.
    UPROPERTY(BlueprintReadWrite)
    TArray<FTerritoryEntityList> OutEntityLists;
};

UENUM()
enum class EPlayerId : int32
{
    Player0 = 0,
    Player1 = 1,
    Player2 = 2,
    Player3 = 3,
    Player4 = 4,
    Player5 = 5,
    Player6 = 6,
    Player7 = 7,
    Player8 = 8,
    MAX
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnMCTSComplete, int32, Action);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAIModelActionComplete, int32, Action);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnTrainingComplete);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnTrainingProgress, float, ProgressPercent, FTrainingStats, Stats);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnSamplesFlushedDelegate);


UCLASS()
class AXISALLIESAI_API UAIManager : public UObject
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "AI|Debug")
    TArray<FTerritoryEntityList> DebugSimulateTransition(
        const TArray<float>& InState, int32 InPhaseId, int32 InPlayerId, int32 Action,
        int32 PendingA, int32 PendingB, int32 PendingC, int32 PendingD,
        const TArray<FTerritoryEntityList>& InEntityLists);

    UPROPERTY(BlueprintAssignable, Category = "AI|Training")
    FOnSamplesFlushedDelegate OnSamplesFlushed;
    //
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    void BeginStagingSession()
    {
        UAI_ReplayBufferManager::Get().BeginGameSession();
        CurrentGameSampleCount = 0;
        bEpisodeTerminalLocked = false;
        LastSampledActionRootIndex = INDEX_NONE;
        bHasEmittedRootSampleThisStep = false;
        DiscardedForcedMoveCount = 0;
        DiscardedRejectedCount = 0;
    }
    // Set pending action context when loading
    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    void SetPendingActionContext(
        int32 PendingActionA,
        int32 PendingActionB,
        int32 PendingActionC,
        int32 PendingActionD)
    {
        PendingRootActionContext.PendingActionA = PendingActionA;
        PendingRootActionContext.PendingActionB = PendingActionB;
        PendingRootActionContext.PendingActionC = PendingActionC;
        PendingRootActionContext.PendingActionD = PendingActionD;
        PendingRootActionContext.bIsValid = true;
    }
    // Gets total number of samples in current staging file
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    int32 GetSamplesInStagingFile() const
    {
        return UAI_ReplayBufferManager::Get().GetSamplesInStagingFile();
    }
    // Clears the MCTS tree after dice rolls
    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    void InvalidateMCTSTree()
    {
        ResetMCTS();
        Tree = FMCTSTree();
    }
    // Clears in-memory samples without saving them
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    void ClearSamplesInMemory()
    {
        UAI_ReplayBufferManager::Get().ClearBuffer();
    }
    // Returns number of samples currently in memory
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    int32 GetSamplesInMemory() const
    {
        return UAI_ReplayBufferManager::Get().GetSamplesInMemory();
    }

    // Returns all staging session file names without extension
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    TArray<FString> GetStagingFileNames() const
    {
        return UAI_ReplayBufferManager::Get().GetStagingFileNames();
    }

    // Sets the current staging session name without loading
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    void SetStagingSessionName(const FString& SessionName)
    {
        UAI_ReplayBufferManager::Get().SetStagingSessionName(SessionName);
    }

    // Sets the current staging session name and confirms file exists
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    bool LoadStagingSession(const FString& SessionName)
    {
        return UAI_ReplayBufferManager::Get().LoadStagingSession(SessionName);
    }

    // Immediately flushes all in-memory samples to current staging file
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    bool FlushInMemorySamplesToStaging()
    {
        return UAI_ReplayBufferManager::Get().FlushPartialToDisk();
    }

    // Deletes the staging file for the given session name
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    bool DeleteStagingFile(const FString& SessionName)
    {
        return UAI_ReplayBufferManager::Get().DeleteStagingFile(SessionName);
    }
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    int32 GetSamplesOnDisk() const
    {
        return UAI_ReplayBufferManager::Get().GetSamplesOnDisk();
    }
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    int32 GetDiscardedSampleCount(int32& OutForcedMoves, int32& OutRejected) const
    {
        OutForcedMoves = DiscardedForcedMoveCount;
        OutRejected = DiscardedRejectedCount;
        return OutForcedMoves + OutRejected;
    }
    // Fired on game thread when MCTS completes; Action=-1 if cancelled or failed
    UPROPERTY(BlueprintAssignable, Category = "AI|MCTS")
    FOnMCTSComplete OnMCTSComplete;

    // Fired on game thread when GetActionFromAIModel completes; Action=-1 if failed
    UPROPERTY(BlueprintAssignable, Category = "AI|Models")
    FOnAIModelActionComplete OnAIModelActionComplete;

    // Fired on game thread when training pipeline completes
    UPROPERTY(BlueprintAssignable, Category = "AI|Training")
    FOnTrainingComplete OnTrainingComplete;

    // Fired on game thread with progress 0.0-1.0 during training
    UPROPERTY(BlueprintAssignable, Category = "AI|Training")
    FOnTrainingProgress OnTrainingProgress;

    // Cancel an in-progress MCTS search; result delivered via OnMCTSComplete(-1)
    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    void CancelMCTS();

    // Returns true if MCTS is currently running on background thread
    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    bool IsMCTSRunning() const;

    // Returns true if training pipeline is currently running on background thread
    UFUNCTION(BlueprintCallable, Category = "AI|Training")
    bool IsTrainingRunning() const;

    static constexpr int32 NumPlayers = static_cast<int32>(EPlayerId::MAX);

    UFUNCTION(BlueprintCallable, Category = "AI|Models")
    bool EnsureModelsExist();

    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    void BeginMCTS(
        const TArray<float>& GameState,
        int32 PhaseId,
        int32 PlayerId,
        UObject* InGameState,
        const TArray<FTerritoryEntityList>& InEntityLists
    );

    UFUNCTION(BlueprintCallable, Category = "AI|Models")
    int32 GetActionFromAIModel(
        const TArray<float>& GameState,
        int32 PhaseId,
        int32 PlayerId,
        const TArray<FTerritoryEntityList>& InEntityLists,
        int32 PendingActionA,
        int32 PendingActionB,
        int32 PendingActionC,
        int32 PendingActionD);

    // ------------------------------------------------------------------
    // Called by C++ to get the legal mask for a node.
    // For chain-intermediate phases, PendingActionA/PendingActionB carry
    // prior sub-decisions. For standalone or chain-first phases,
    // bHasPendingContext is false and both pending values are INDEX_NONE.
    // InEntityLists contains the current entity state for all territories,
    // read from the node being expanded.
    // Only OutLegalActionMask and bIsTerminal are read from the result.
    // ------------------------------------------------------------------
    UFUNCTION(BlueprintImplementableEvent, Category = "AI|GameState")
    FApplyActionResult GetLegalActionMask(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        bool bHasPendingContext,
        int32 PendingActionA,
        int32 PendingActionB,
        int32 PendingActionC,
        int32 PendingActionD,
        const TArray<FTerritoryEntityList>& InEntityLists);

    // Called by MCTS after dice are sampled and hits are encoded into the
    // node feature buffer. Returns the legal action mask over the casualty
    // assignment action space for the given state.
    UFUNCTION(BlueprintImplementableEvent, Category = "AI|Combat")
    FApplyActionResult GetCasualtyAssignmentMask(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        const TArray<FTerritoryEntityList>& InEntityLists
    );

    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    int32 CreateRootNode(
        const TArray<float>& GameState,
        int32 PhaseId,
        int32 PlayerId,
        const TArray<FTerritoryEntityList>& InEntityLists
    );

    void ExpandNode(int32 NodeIndex, int32 Action);
    void ApplyInferenceResultToNode(
        int32 NodeIndex,
        const TArray<float>& Policy,
        const TArray<float>& Value);
    void FlushInferenceBatch();
    void ApplyRootDirichletNoise(FMCTSNode& Root, float Epsilon, float Alpha);

    // Resets the MCTS tree and clears the inference queue.
    // All entity lists stored on nodes are freed when the tree is reset.
    void ResetMCTS();
    void FinalizeMCTS();

    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    int32 GetActionFromMCTS(
        const TArray<float>& GameState,
        int32 PhaseId,
        int32 PlayerId,
        UObject* InGameState,
        const TArray<FTerritoryEntityList>& InEntityLists
    );

    UPROPERTY()
    FMCTSContext MCTSContext;

    float GetMCTSTemperature() const;
    bool IsEvaluationMode() const;
    bool IsTrainingMode() const;
    bool ValidateMCTSRuntimeInvariants(const FMCTSNode& Node);

    int32 FindExistingNodeByState(
        const TArray<float>& GameState,
        int32 PhaseId,
        int32 PlayerId) const;

    bool ValidateInferenceOutputsNumerical(
        int32 BatchSize,
        const TArray<float>& OutHead2,
        const TArray<float>& OutHead3,
        const TArray<float>& OutHead7,
        const TArray<float>& OutHead10,
        const TArray<float>& OutHead13,
        const TArray<float>& OutHead14,
        const TArray<float>& OutHead20,
        const TArray<float>& OutHead49,
        const TArray<float>& OutHead128,
        const TArray<float>& OutHead202,
        const TArray<float>& OutHead330,
        const TArray<float>& OutHead6581,
        const TArray<float>& OutHead988,
        const TArray<float>& OutValue);

    int32 PromoteRootToChild(int32 CurrentRootIndex, int32 Action);
    int32 GetActionSizeForPhase(int32 PhaseId);
    int32 GetAdaptiveSimulationCount(int32 RootIndex);
    void ValidatePhaseRoutingInvariant(int32 NodeIndex);

    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    bool ExportReplayBuffer();

    UFUNCTION(BlueprintCallable, Category = "AI|SmokeTest")
    bool RunEndToEndInferenceSmokeTest();

    const FMCTSTree& GetTreeForReplay() const;
    TArray<float> BuildRootPolicyTarget(int32 RootIndex, float Temperature);

    void BeginAIArenaGame();

    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    bool EndAIArenaGame(const TArray<float>& FinalOutcomeValues);

    UFUNCTION(BlueprintCallable, Category = "AI|MCTS")
    void FinalizeMCTSTrainingPipeline();

    // ------------------------------------------------------------------
    // State buffer extraction helpers.
    // Node features: first NUM_TERRITORIES * NODE_FEATURE_COUNT floats.
    // Global features: next GLOBAL_FEATURE_COUNT floats.
    // Entity tensors are not extracted from the state buffer — they are
    // read directly from FMCTSNode::EntityLists.
    // ------------------------------------------------------------------
    TArray<float> ExtractNodeFeatures(const TArray<float>& StateBuffer);
    TArray<float> ExtractGlobalFeatures(const TArray<float>& StateBuffer);

    // ------------------------------------------------------------------
    // Assembles the padded entity tensor for inference from a node's
    // EntityLists. Outputs flat arrays suitable for passing to RDGCache:
    //   OutEntityTensor: [N * E * UNIT_ENTITY_FEATURE_COUNT] floats
    //   OutEntityCounts: [N] floats
    // where N = NUM_TERRITORIES, E = MAX_UNIT_ENTITIES_PER_NODE.
    // ------------------------------------------------------------------
    void AssembleEntityTensor(
        const TArray<FTerritoryEntityList>& EntityLists,
        TArray<float>& OutEntityTensor,
        TArray<float>& OutEntityCounts
    );

    UFUNCTION(BlueprintCallable, Category = "AI|SmokeTest")
    bool RunCombatResolutionTest(int32 MaxRounds);

    UFUNCTION(BlueprintCallable, Category = "AI|SmokeTest")
    bool RunFullMCTSSystemTest();

    // ------------------------------------------------------------------
    // SimulateTransition — BlueprintImplementableEvent.
    // For chain-final phases, Action is the final sub-decision;
    // PendingActionA/PendingActionB carry prior sub-decisions.
    // For standalone phases, bHasPendingContext is false.
    // Blueprint populates OutEntityLists in the returned result with
    // the full updated entity state after applying the action.
    // C++ stores OutEntityLists on the child node.
    // ------------------------------------------------------------------
    UFUNCTION(BlueprintImplementableEvent, Category = "AI|GameState")
    FApplyActionResult SimulateTransition(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        int32 Action,
        bool bHasPendingContext,
        int32 PendingActionA,
        int32 PendingActionB,
        int32 PendingActionC,
        int32 PendingActionD,
        const TArray<FTerritoryEntityList>& InEntityLists);

    UFUNCTION(BlueprintCallable, Category = "AI|Territory")
    bool GetTerritoryInfo(
        int32 NodeIndex,
        FString& OutName,
        TArray<int32>& OutConnectedNodes);

    UFUNCTION(BlueprintCallable, Category = "AI|State")
    bool LoadInitialGameState(
        TArray<float>& OutStateBuffer,
        TArray<FTerritoryEntityList>& OutEntityLists,
        int32& OutPhaseId,
        int32& OutActingPlayer,
        int32& OutRound);

private:
    FPendingActionContext PendingRootActionContext;

    bool SampleCombatDice(int32 NodeIndex);
    bool SimulateCombatRoundInternal(int32 NodeIndex);

    FApplyActionResult StubGetCasualtyAssignmentMask(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        const TArray<FTerritoryEntityList>& InEntityLists);

    int32 GetNodeDepth(int32 NodeIndex) const;

    FApplyActionResult StubSimulateTransition(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        int32 Action,
        int32 NodeIndex,
        const TArray<FTerritoryEntityList>& InEntityLists);

    FApplyActionResult StubGetLegalActionMask(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        const TArray<FTerritoryEntityList>& InEntityLists);

    static constexpr int32 StubMaxChainDepth = 20;
    static constexpr int32 ReplayBufferFlushThreshold = 50;

    FApplyActionResult Internal_SimulateTransition(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        int32 Action,
        bool bHasPendingContext,
        int32 PendingActionA,
        int32 PendingActionB,
        int32 PendingActionC,
        int32 PendingActionD,
        int32 NodeIndex,
        const TArray<FTerritoryEntityList>& InEntityLists);

    FApplyActionResult Internal_GetLegalActionMask(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        bool bHasPendingContext,
        int32 PendingActionA,
        int32 PendingActionB,
        int32 PendingActionC,
        int32 PendingActionD,
        int32 NodeIndex,
        const TArray<FTerritoryEntityList>& InEntityLists);

    FApplyActionResult Internal_GetCasualtyAssignmentMask(
        const TArray<float>& InState,
        int32 InPhaseId,
        int32 InPlayerId,
        const TArray<FTerritoryEntityList>& InEntityLists);

    bool bUseStubTransitionFunctions = false;

    // Clears entity lists on all nodes in the tree except the node at
    // ExceptIndex. Called during re-rooting to free stale entity state.
    void ClearAllEntityListsExcept(int32 ExceptIndex);

    int32 SelectFinalActionFromVisits(int32 RootIndex, float Temperature);
    void EvaluateNode(int32 NodeIndex);
    int32 SelectActionPUCT(int32 NodeIndex, float C);
    void Backpropagate(int32 NodeIndex, int32 RootPlayerId);

    int32 LastSampledActionRootIndex = INDEX_NONE;
    bool  bHasEmittedRootSampleThisStep = false;
    int32 CurrentGameSampleCount = 0;
    int32 CurrentEpisodeId = 0;
    bool  bEpisodeTerminalLocked = false;

    bool SampleTechDice(int32 NodeIndex, int32 NumDice, int32 PlayerId);
    bool SampleKamikazeDice(int32 NodeIndex);
    bool SampleBombardmentDice(int32 NodeIndex);
    bool SampleStrategicBombingDice(int32 NodeIndex);
    bool SampleSubmarineSurpriseDice(int32 NodeIndex);

private:
    TAtomic<int32> DiscardedForcedMoveCount{ 0 };
    TAtomic<int32> DiscardedRejectedCount{ 0 };

    TAtomic<bool>   bCancelRequested{ false };
    FThreadSafeBool bMCTSRunning{ false };
    FThreadSafeBool bTrainingRunning{ false };
    TFuture<void>   MCTSFuture;
    TFuture<void>   TrainingFuture;

    UPROPERTY()
    TObjectPtr<UObject> GameStateRef;

    UPROPERTY()
    FMCTSTree Tree;

    UPROPERTY()
    UAI_RDGCache* RDGCache = nullptr;

    UPROPERTY()
    UAI_InferenceQueue* InferenceQueue;

    TSet<int32> InFlightNodes;
    bool bRootFinalized = false;

    UPROPERTY()
    TArray<float> CachedFinalOutcomeValues;

    bool bTerritoryGraphLoaded = false;
    TMap<int32, FString>       CachedNodeNames;
    TMap<int32, TArray<int32>> CachedAdjacency;

    FRandomStream MCTSRandStream;
};