// UAI_InferenceQueue.h
#pragma once
#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "UAI_InferenceQueue.generated.h"

USTRUCT()
struct FInferenceCacheKey
{
    GENERATED_BODY()

    uint32 StateHash = 0;
    int32  PhaseId = 0;
    int32  PlayerId = 0;
    int32  ActionSize = 0;

    bool operator==(const FInferenceCacheKey& Other) const
    {
        return StateHash == Other.StateHash &&
            PhaseId == Other.PhaseId &&
            PlayerId == Other.PlayerId &&
            ActionSize == Other.ActionSize;
    }
};

FORCEINLINE uint32 GetTypeHash(const FInferenceCacheKey& Key)
{
    uint32 Hash = Key.StateHash;
    Hash = HashCombine(Hash, (uint32)Key.PhaseId);
    Hash = HashCombine(Hash, (uint32)Key.PlayerId);
    Hash = HashCombine(Hash, (uint32)Key.ActionSize);
    return Hash;
}

USTRUCT()
struct FInferenceRequest
{
    GENERATED_BODY()

    // Unique request identifier for RDG tracking
    int32  RequestId = 0;

    // Core game identifiers
    int32  PhaseId = 0;
    int32  PlayerId = 0;
    int32  ActionSize = 0;
    uint32 StateHash = 0;
    int32  NodeIndex = -1;
    int32  ActionIndex = -1;

    // =========================
    // GRAPH TRANSFORMER INPUTS
    // =========================

    // Territory node features: NUM_TERRITORIES * NODE_FEATURE_COUNT floats
    TArray<float> NodeFeatures;

    // Board-wide global features: GLOBAL_FEATURE_COUNT floats
    TArray<float> GlobalFeatures;

    // Unit entity tensor: NUM_TERRITORIES * MAX_UNIT_ENTITIES_PER_NODE *
    // UNIT_ENTITY_FEATURE_COUNT floats, laid out as [territory][entity][feature].
    // Assembled by UAIManager::AssembleEntityTensor from FTerritoryEntityList.
    // Zero-padded for territories with fewer than MAX_UNIT_ENTITIES_PER_NODE entities.
    TArray<float> EntityTensor;

    // Valid entity count per territory: NUM_TERRITORIES floats.
    // EntityCounts[t] = number of valid (non-padding) entities at territory t.
    // Used by the entity encoder's additive padding mask in model.py.
    TArray<float> EntityCounts;

    // =========================
    // OUTPUT STORAGE
    // =========================

    TArray<float> PolicyOutput;
    TArray<float> ValueOutput;
    bool bCompleted = false;

    FInferenceRequest() {}
};

UCLASS()
class AXISALLIESAI_API UAI_InferenceQueue : public UObject
{
    GENERATED_BODY()

public:
    int32 GetQueueSize() const { return Queue.Num(); }
    void Enqueue(const FInferenceRequest& Request);
    void DequeueAll(TArray<FInferenceRequest>& OutRequests);
    void Reset();
    void MarkCompleted(const FInferenceCacheKey& Key);
    bool TryGetCache(
        const FInferenceCacheKey& Key,
        TPair<TArray<float>, TArray<float>>& Out);
    void StoreCache(
        const FInferenceCacheKey& Key,
        const TArray<float>& Policy,
        const TArray<float>& Value);
    void Clear();

    FInferenceCacheKey BuildKey(
        const TArray<float>& State,
        int32 PhaseId,
        int32 PlayerId,
        int32 ActionSize);

    FInferenceCacheKey BuildGraphKey(
        const TArray<float>& NodeFeatures,
        const TArray<float>& GlobalFeatures,
        int32 PhaseId,
        int32 PlayerId,
        int32 ActionSize);

private:
    TArray<FInferenceRequest>                                        Queue;
    TMap<FInferenceCacheKey, TPair<TArray<float>, TArray<float>>>    Cache;
    TSet<FInferenceCacheKey>                                         InFlight;
};