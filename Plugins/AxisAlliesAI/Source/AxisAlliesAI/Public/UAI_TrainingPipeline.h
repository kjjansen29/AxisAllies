#pragma once

#include "CoreMinimal.h"
#include "UAI_TrainingPipeline.generated.h"

USTRUCT()
struct FTrainingTensorBatch
{
    GENERATED_BODY()

    int32 PhaseId = 0;

    TArray<TArray<float>> States;

    // Aligned with PhaseId policy head
    TArray<TArray<float>> PolicyTargets;

    // Aligned with model.py value head ordering
    TArray<TArray<float>> ValueTargets;

    TArray<int32> PlayerIds;
};

class AXISALLIESAI_API UAI_TrainingPipeline
{
public:
    bool RunPythonTrainingStep(
        const FString& PythonExecutablePath,
        const FString& ModelScriptPath,
        const FString& DatasetPath,
        const FString& OutputModelDir);
};