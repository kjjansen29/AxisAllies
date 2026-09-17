#include "UAI_TrainingPipeline.h"
#include "UAI_ReplayBufferManager.h"
#include "AIInferenceSpec.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"

bool UAI_TrainingPipeline::RunPythonTrainingStep(
    const FString& PythonExecutablePath,
    const FString& ModelScriptPath,
    const FString& DatasetPath,
    const FString& OutputModelDir)
{
    if (PythonExecutablePath.IsEmpty() ||
        ModelScriptPath.IsEmpty() ||
        DatasetPath.IsEmpty() ||
        OutputModelDir.IsEmpty())
    {
        UE_LOG(LogTemp, Error, TEXT("[TrainingPipeline] One or more arguments are empty."));
        return false;
    }

    IFileManager::Get().MakeDirectory(*OutputModelDir, true);

    const FString Arguments = FString::Printf(
        TEXT("\"%s\" --dataset \"%s\" --out_dir \"%s\" --batch_size 64 --steps 500"),
        *ModelScriptPath,
        *DatasetPath,
        *OutputModelDir
    );

    UE_LOG(LogTemp, Warning, TEXT("[TrainingPipeline] Invoking: %s %s"),
        *PythonExecutablePath, *Arguments);

    int32 ReturnCode = -1;
    FString StdOut;
    FString StdErr;

    const bool bProcessSuccess = FPlatformProcess::ExecProcess(
        *PythonExecutablePath,
        *Arguments,
        &ReturnCode,
        &StdOut,
        &StdErr
    );

    UE_LOG(LogTemp, Warning, TEXT("[TrainingPipeline] bProcessSuccess=%s ReturnCode=%d"),
        bProcessSuccess ? TEXT("true") : TEXT("false"), ReturnCode);

    if (!StdOut.IsEmpty())
    {
        UE_LOG(LogTemp, Warning, TEXT("[TrainingPipeline] stdout:\n%s"), *StdOut);
    }

    if (!StdErr.IsEmpty())
    {
        UE_LOG(LogTemp, Error, TEXT("[TrainingPipeline] stderr:\n%s"), *StdErr);
    }

    if (!bProcessSuccess || ReturnCode != 0)
    {
        UE_LOG(LogTemp, Error,
            TEXT("[TrainingPipeline] Python process failed. bProcessSuccess=%s ReturnCode=%d"),
            bProcessSuccess ? TEXT("true") : TEXT("false"), ReturnCode);
        return false;
    }

    const FString PtModelPath = GetPTPath();
    const FString OnnxModelPath = GetONNXPath();

    if (!FPaths::FileExists(PtModelPath))
    {
        UE_LOG(LogTemp, Error, TEXT("[TrainingPipeline] .pt not found at: %s"), *PtModelPath);
        return false;
    }

    if (!FPaths::FileExists(OnnxModelPath))
    {
        UE_LOG(LogTemp, Error, TEXT("[TrainingPipeline] .onnx not found at: %s"), *OnnxModelPath);
        return false;
    }

    if (IFileManager::Get().FileSize(*PtModelPath) <= 0 ||
        IFileManager::Get().FileSize(*OnnxModelPath) <= 0)
    {
        UE_LOG(LogTemp, Error, TEXT("[TrainingPipeline] Artifact files are empty after training."));
        return false;
    }

    return true;
}