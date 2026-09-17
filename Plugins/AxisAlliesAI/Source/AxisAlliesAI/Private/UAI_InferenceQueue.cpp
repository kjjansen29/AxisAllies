#include "UAI_InferenceQueue.h"
#include "UAI_RDGCache.h"
#include "AIInferenceSpec.h"

void UAI_InferenceQueue::DequeueAll(TArray<FInferenceRequest>& OutRequests)
{
    OutRequests = MoveTemp(Queue);
    Queue.Reset();
    // InFlight must NOT be reset here in corrected lifecycle model.
    // Completion is controlled explicitly via MarkCompleted per key.
}

void UAI_InferenceQueue::Enqueue(const FInferenceRequest& Request)
{
    // ------------------------------------------------------------
    // GRAPH TRANSFORMER CONTRACT ENFORCEMENT
    // ------------------------------------------------------------
    if (Request.ActionSize <= 0)
        return;

    if (Request.NodeFeatures.Num() != NUM_TERRITORIES * NODE_FEATURE_COUNT)
        return;

    if (Request.GlobalFeatures.Num() != GLOBAL_FEATURE_COUNT)
        return;

    if (Request.PhaseId < 0 || Request.PhaseId >= PHASE_COUNT)
        return;

    // Entity tensor must be exactly NUM_TERRITORIES * MAX_UNIT_ENTITIES_PER_NODE
    // * UNIT_ENTITY_FEATURE_COUNT floats, or empty (empty is accepted for nodes
    // where AssembleEntityTensor has not yet been called, e.g. during testing).
    const int32 ExpectedEntityTensorSize =
        NUM_TERRITORIES * MAX_UNIT_ENTITIES_PER_NODE * UNIT_ENTITY_FEATURE_COUNT;
    if (Request.EntityTensor.Num() != 0 &&
        Request.EntityTensor.Num() != ExpectedEntityTensorSize)
    {
        return;
    }

    // Entity counts must be exactly NUM_TERRITORIES floats, or empty.
    if (Request.EntityCounts.Num() != 0 &&
        Request.EntityCounts.Num() != NUM_TERRITORIES)
    {
        return;
    }

    // ------------------------------------------------------------
    // KEY CONSTRUCTION (GRAPH-AWARE ONLY)
    // Entity tensor is intentionally excluded from the cache key.
    // The cache key is derived from node and global features only,
    // matching the assumption that entity state is fully reflected
    // in the node features for cache-hit purposes. If two requests
    // share identical node and global features, their entity state
    // is expected to be identical as well.
    // ------------------------------------------------------------
    FInferenceCacheKey Key;
    Key.StateHash = BuildGraphKey(
        Request.NodeFeatures,
        Request.GlobalFeatures,
        Request.PhaseId,
        Request.PlayerId,
        Request.ActionSize
    ).StateHash;
    Key.PhaseId = Request.PhaseId;
    Key.PlayerId = Request.PlayerId;
    Key.ActionSize = Request.ActionSize;

    // ------------------------------------------------------------
    // IN-FLIGHT GUARD
    // ------------------------------------------------------------
    if (InFlight.Contains(Key))
        return;

    InFlight.Add(Key);
    Queue.Add(Request);
}

void UAI_InferenceQueue::Reset()
{
    Queue.Reset();
    Cache.Reset();
    InFlight.Reset();
}

bool UAI_InferenceQueue::TryGetCache(
    const FInferenceCacheKey& Key,
    TPair<TArray<float>, TArray<float>>& Out)
{
    const TPair<TArray<float>, TArray<float>>* Found = Cache.Find(Key);
    if (!Found)
        return false;
    Out = *Found;
    return true;
}

void UAI_InferenceQueue::StoreCache(
    const FInferenceCacheKey& Key,
    const TArray<float>& Policy,
    const TArray<float>& Value)
{
    if (Value.Num() == 0)
        return;

    Cache.Add(Key, TPair<TArray<float>, TArray<float>>(Policy, Value));

    if (InFlight.Contains(Key))
        InFlight.Remove(Key);
}

void UAI_InferenceQueue::Clear()
{
    Queue.Reset();
    Cache.Reset();
    InFlight.Empty();
}

FInferenceCacheKey UAI_InferenceQueue::BuildKey(
    const TArray<float>& State,
    int32 PhaseId,
    int32 PlayerId,
    int32 ActionSize)
{
    FInferenceCacheKey Key;
    uint32 StateHash = 0;
    if (State.Num() > 0)
    {
        StateHash = FCrc::MemCrc32(
            State.GetData(),
            State.Num() * sizeof(float));
    }
    Key.StateHash = StateHash;
    Key.PhaseId = PhaseId;
    Key.PlayerId = PlayerId;
    Key.ActionSize = ActionSize;
    return Key;
}

void UAI_InferenceQueue::MarkCompleted(const FInferenceCacheKey& Key)
{
    TPair<TArray<float>, TArray<float>> CachedResult;
    const bool bHasCache = TryGetCache(Key, CachedResult);
    if (!bHasCache)
        return;

    const bool bValidPolicy = CachedResult.Key.Num() > 0;
    const bool bValidValue = CachedResult.Value.Num() > 0;
    if (!bValidPolicy || !bValidValue)
        return;

    InFlight.Remove(Key);
}

FInferenceCacheKey UAI_InferenceQueue::BuildGraphKey(
    const TArray<float>& NodeFeatures,
    const TArray<float>& GlobalFeatures,
    int32 PhaseId,
    int32 PlayerId,
    int32 ActionSize)
{
    FInferenceCacheKey Key;
    uint32 GraphHash = 0;

    if (NodeFeatures.Num() > 0)
    {
        GraphHash = FCrc::MemCrc32(
            NodeFeatures.GetData(),
            NodeFeatures.Num() * sizeof(float));
    }

    if (GlobalFeatures.Num() > 0)
    {
        const uint32 GlobalHash = FCrc::MemCrc32(
            GlobalFeatures.GetData(),
            GlobalFeatures.Num() * sizeof(float));
        GraphHash = HashCombine(GraphHash, GlobalHash);
    }

    Key.StateHash = GraphHash;
    Key.PhaseId = PhaseId;
    Key.PlayerId = PlayerId;
    Key.ActionSize = ActionSize;
    return Key;
}