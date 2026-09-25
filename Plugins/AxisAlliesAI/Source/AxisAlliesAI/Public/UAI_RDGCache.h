// UAI_RDGCache.h
#pragma once
#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "NNEModelData.h"
#include "NNERuntimeRDG.h"
#include "NNERuntimeCPU.h"
#include "UAIModelArtifactManager.h"
#include "UAI_InferenceQueue.h"
#include "MCTSNode.h"
#include "UAI_RDGCache.generated.h"

UCLASS()
class AXISALLIESAI_API UAI_RDGCache : public UObject
{
    GENERATED_BODY()

public:
    // ----------------------------------------------------------------
    // GPU inference timing, accumulated across RunInference calls.
    // Reset and reported by UAIManager once per search.
    // ----------------------------------------------------------------
    struct FInferenceTimingStats
    {
        double PrepSeconds = 0.0;            // flattening inputs on the calling thread
        double GameThreadWaitSeconds = 0.0;  // waiting for the game thread to start the work
        double UploadSeconds = 0.0;          // first render flush: buffer creation + upload
        double ExecuteSeconds = 0.0;         // second render flush: model run + GPU idle wait
        double ReadbackSeconds = 0.0;        // copying results back from the GPU
    };
    static FInferenceTimingStats TimingStats;

    // ----------------------------------------------------------------
    // Inference runtime used for MCTS and GetActionFromAIModel.
    //   TEXT("NNERuntimeORTCpu")  — runs on the processor, on the calling
    //                               (search) thread. No game-thread or GPU sync.
    //   TEXT("NNERuntimeORTDml")  — runs on the GPU through the render graph.
    // Change this one value to switch; everything else follows it.
    // ----------------------------------------------------------------
    static constexpr const TCHAR* InferenceRuntimeName = TEXT("NNERuntimeORTDml");

    // RuntimeName ending in "Cpu" selects the CPU path, otherwise the GPU path.
    bool Initialize(const FString& RuntimeName = InferenceRuntimeName);
    void Reset();

    bool IsUsingCPU() const { return bUseCPU; }

    // RunInference now accepts entity tensors alongside node and global features.
    //
    // EntityTensorBatch: one flat array per sample in the batch.
    //   Each array has NUM_TERRITORIES * MAX_UNIT_ENTITIES_PER_NODE *
    //   UNIT_ENTITY_FEATURE_COUNT floats, laid out as [territory][entity][feature].
    //   Assembled by UAIManager::AssembleEntityTensor from FTerritoryEntityList.
    //
    // EntityCountsBatch: one flat array per sample.
    //   Each array has NUM_TERRITORIES floats — the count of valid entities
    //   at each territory, used by the entity encoder's padding mask.
    bool RunInference(
        const TArray<TArray<float>>& NodeFeaturesBatch,
        const TArray<TArray<float>>& GlobalFeaturesBatch,
        const TArray<TArray<float>>& EntityTensorBatch,
        const TArray<TArray<float>>& EntityCountsBatch,
        const TArray<float>& PhaseIds,
        TArray<float>& OutHead2,
        TArray<float>& OutHead3,
        TArray<float>& OutHead7,
        TArray<float>& OutHead10,
        TArray<float>& OutHead13,
        TArray<float>& OutHead14,
        TArray<float>& OutHead20,
        TArray<float>& OutHead49,
        TArray<float>& OutHead128,
        TArray<float>& OutHead202,
        TArray<float>& OutHead330,
        TArray<float>& OutHead6581,
        TArray<float>& OutHead988,
        TArray<float>& OutValue
    );

    bool ValidatePendingModelSwap();
    bool ApplyModelSwap(const FString& InPendingONNXPath);
    void TickModelHotSwap();
    void FlushInferenceGraph();
    bool ReinitializeModelFromONNX(const FString& InONNXPath);
    void ResetInferenceState();
    bool RequestModelSwap(const FString& NewONNXPath, const FString& NewPTPath);

    UPROPERTY()
    TSoftObjectPtr<UNNEModelData> ModelDataAsset;

    UPROPERTY()
    TObjectPtr<UNNEModelData> ModelData;

private:
    FCriticalSection ModelInstanceMutex;

    UNNEModelData* LoadModelDataFromONNX(const FString& InONNXPath);
    bool ValidateModelArtifactSignature(const FString& InONNXPath);
    bool RebuildRDGModelFromData(UNNEModelData* LoadedModelData);
    bool RebuildCPUModelFromData(UNNEModelData* LoadedModelData);
    void ReleaseCurrentModel();

    // Same parameters as RunInference; RunInference forwards to one of these.
    bool RunInferenceGPU(
        const TArray<TArray<float>>& NodeFeaturesBatch,
        const TArray<TArray<float>>& GlobalFeaturesBatch,
        const TArray<TArray<float>>& EntityTensorBatch,
        const TArray<TArray<float>>& EntityCountsBatch,
        const TArray<float>& PhaseIds,
        TArray<float>& OutHead2, TArray<float>& OutHead3, TArray<float>& OutHead7,
        TArray<float>& OutHead10, TArray<float>& OutHead13, TArray<float>& OutHead14,
        TArray<float>& OutHead20, TArray<float>& OutHead49, TArray<float>& OutHead128,
        TArray<float>& OutHead202, TArray<float>& OutHead330, TArray<float>& OutHead6581,
        TArray<float>& OutHead988, TArray<float>& OutValue);

    bool RunInferenceCPU(
        const TArray<TArray<float>>& NodeFeaturesBatch,
        const TArray<TArray<float>>& GlobalFeaturesBatch,
        const TArray<TArray<float>>& EntityTensorBatch,
        const TArray<TArray<float>>& EntityCountsBatch,
        const TArray<float>& PhaseIds,
        TArray<float>& OutHead2, TArray<float>& OutHead3, TArray<float>& OutHead7,
        TArray<float>& OutHead10, TArray<float>& OutHead13, TArray<float>& OutHead14,
        TArray<float>& OutHead20, TArray<float>& OutHead49, TArray<float>& OutHead128,
        TArray<float>& OutHead202, TArray<float>& OutHead330, TArray<float>& OutHead6581,
        TArray<float>& OutHead988, TArray<float>& OutValue);

    TArray64<uint8> ModelBytes;
    TSharedPtr<UE::NNE::IModelRDG>         ModelRDG;
    TSharedPtr<UE::NNE::IModelInstanceRDG> ModelInstance;

    // CPU path
    TSharedPtr<UE::NNE::IModelCPU>         ModelCPU;
    TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstanceCPU;
    FCriticalSection                       CPUInferenceMutex;
    int32                                  CPUShapeBatchSize = -1;
    bool                                   bUseCPU = false;

    int32 LastBatchSize = -1;

    FModelArtifactSignature LastAcceptedArtifactSignature;
    FModelArtifactSignature PendingArtifactSignature;

    FString PendingONNXPath;
    FString PendingPTPath;
    bool    bPendingModelSwap = false;
    bool    bModelInitialized = false;
};