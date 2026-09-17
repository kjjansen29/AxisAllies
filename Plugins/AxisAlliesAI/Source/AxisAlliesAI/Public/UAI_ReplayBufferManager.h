#pragma once
#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "MCTSNode.h"
#include "AIInferenceSpec.h"
#include "UAI_TrainingPipeline.h"
#include "UAI_ReplayBufferManager.generated.h"

USTRUCT()
struct FMCTSTrainingSample
{
    GENERATED_BODY()

    // ------------------------------------------------------------
    // GRAPH TRANSFORMER INPUTS (MODEL INPUT)
    // ------------------------------------------------------------

    // Territory node features: NUM_TERRITORIES * NODE_FEATURE_COUNT floats
    TArray<float> NodeFeatures;

    // Board-wide global features: GLOBAL_FEATURE_COUNT floats
    TArray<float> GlobalFeatures;

    // Unit entity lists: one entry per territory (NUM_TERRITORIES total).
    // Serialized to JSON as "entity_list" in ExportReplayBuffer, matching
    // the format expected by _build_entity_tensors() in model.py:
    //   [{count: N, entities: [[f0..f16], ...]}, ...]
    TArray<FTerritoryEntityList> EntityLists;

    // ------------------------------------------------------------
    // POLICY / VALUE TARGETS (MCTS OUTPUTS)
    // ------------------------------------------------------------

    // Per-policy-head distribution (PhaseId-specific action space)
    TArray<float> PolicyTarget;

    // Per-player value targets: NUM_PLAYERS floats (ordered as model.py output)
    TArray<float> ValueTarget;

    // ------------------------------------------------------------
    // CONTEXT METADATA
    // ------------------------------------------------------------
    int32 PhaseId = 0;
    int32 PlayerId = 0;
    int32 GameId = 0;
    int32 MoveIndex = 0;

    // ------------------------------------------------------------
    // TRAINING SIGNAL
    // ------------------------------------------------------------
    float Reward = 0.0f;

    // Number of visits from MCTS root statistics (for weighting training targets)
    float VisitCount = 0.0f;
};

class AXISALLIESAI_API UAI_ReplayBufferManager
{
public:
    static UAI_ReplayBufferManager& Get();

    void AddSample(const FMCTSTrainingSample& Sample);
    const TArray<FMCTSTrainingSample>& GetBuffer() const;
    int32 GetSampleCount() const;
    void SetMaxCapacity(int32 InMaxCapacity);
    void Clear();

    bool ExportReplayBuffer(const TArray<float>& FinalOutcomeValues);
    void StoreSelfPlaySample(const FMCTSTree& Tree);
    void BeginGameSession();
    void ApplyFinalGameRewardsToReplayBuffer(const TArray<float>& FinalOutcomeValues);
    bool IsValidTrainingSample(const FMCTSTrainingSample& Sample) const;

    int32 GetDiskSampleCount() const;
    int32 GetNewSamplesSinceLastTraining() const;
    void ResetNewSampleCounter();
    // Returns number of samples currently in memory
    int32 GetSamplesInMemory() const;

    // Returns number of samples on disk in TotalReplayBuffer
    int32 GetSamplesOnDisk() const;

    // Flush in-memory buffer to per-episode staging file with
    // placeholder (0.0f) value targets. Clears in-memory buffer.
    // Called every 100 samples from GetActionFromMCTS.
    bool FlushPartialToDisk();

    // Called by EndAIArenaGame. Loads staged samples for episode,
    // applies actual outcome values, writes to TotalReplayBuffer,
    // deletes staging file.
    bool ApplyOutcomeValuesToEpisode(
        const TArray<float>& OutcomeValues);

    void SaveTrainingMetadata() const;           // existing calls with no args
    void SaveTrainingMetadata(int32 EpisodeId) const;  // new calls from EndAIArenaGame

    void LoadTrainingMetadata();                 // existing calls with no args  
    void LoadTrainingMetadata(int32& OutEpisodeId);    // new calls from BeginAIArenaGame

private:
    UAI_ReplayBufferManager() = default;

    void EnforceCapacity(TArray<FMCTSTrainingSample>& InOutBuffer);
    bool ValidateAndNormalizeReplayExport(
        TArray<FMCTSTrainingSample>& InOutTotalBuffer,
        const TArray<FMCTSTrainingSample>& InMemoryBuffer,
        const TArray<float>& FinalOutcomeValues);

    void SerializeEntityList(
        const FMCTSTrainingSample& Sample,
        TSharedPtr<FJsonObject>& SObj) const;

    void DeserializeSample(
        const TSharedPtr<FJsonObject>& SObj,
        FMCTSTrainingSample& Sample) const;

    TArray<FMCTSTrainingSample> LoadTotalBufferFromDisk() const;

    bool SerializeToDisk(
        const TArray<FMCTSTrainingSample>& TotalBuffer,
        const FString& ExportPath) const;

private:
    TArray<FMCTSTrainingSample> Buffer;
    int32 MaxCapacity = 100000;
    int32 NewSamplesSinceLastTraining = 0;
    bool  bMetadataLoaded = false;
};