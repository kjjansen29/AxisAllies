// UAI_RDGCache.h
#pragma once
#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "NNEModelData.h"
#include "NNERuntimeRDG.h"
#include "UAIModelArtifactManager.h"
#include "UAI_InferenceQueue.h"
#include "MCTSNode.h"
#include "UAI_RDGCache.generated.h"

UCLASS()
class AXISALLIESAI_API UAI_RDGCache : public UObject
{
    GENERATED_BODY()

public:
    bool Initialize(const FString& RuntimeName = TEXT("NNERuntimeORTDml"));
    void Reset();

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

    UNNEModelData* ModelData;

private:
    FCriticalSection ModelInstanceMutex;

    UNNEModelData* LoadModelDataFromONNX(const FString& InONNXPath);
    bool ValidateModelArtifactSignature(const FString& InONNXPath);
    bool RebuildRDGModelFromData(UNNEModelData* LoadedModelData);
    void ReleaseCurrentModel();

    TArray64<uint8> ModelBytes;
    TSharedPtr<UE::NNE::IModelRDG>         ModelRDG;
    TSharedPtr<UE::NNE::IModelInstanceRDG> ModelInstance;

    int32 LastBatchSize = -1;

    FModelArtifactSignature LastAcceptedArtifactSignature;
    FModelArtifactSignature PendingArtifactSignature;

    FString PendingONNXPath;
    FString PendingPTPath;
    bool    bPendingModelSwap = false;
    bool    bModelInitialized = false;
};