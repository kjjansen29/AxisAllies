#include "UAIModelArtifactManager.h"
#include "Misc/FileHelper.h"
#include "Misc/SecureHash.h"
#include "HAL/PlatformFilemanager.h"

FString UAIModelArtifactManager::ComputeFileHash(const FString& FilePath)
{
    TArray<uint8> FileData;

    if (!FFileHelper::LoadFileToArray(FileData, *FilePath))
    {
        ensureMsgf(false, TEXT("Failed to load artifact for hashing: %s"), *FilePath);
        return FString();
    }

    FSHAHash Hash = FSHA1::HashBuffer(FileData.GetData(), FileData.Num());

    return Hash.ToString();
}

FString UAIModelArtifactManager::ComputeUnifiedArtifactHash(const FString& PTPath, const FString& ONNXPath)
{
    const FString PTHash = ComputeFileHash(PTPath);
    const FString ONNXHash = ComputeFileHash(ONNXPath);

    if (PTHash.IsEmpty() || ONNXHash.IsEmpty())
    {
        ensureMsgf(false, TEXT("Failed unified hash generation due to missing artifact data"));
        return FString();
    }

    const FString Combined = PTHash + TEXT("|") + ONNXHash;

    FSHAHash UnifiedHash = FSHA1::HashBuffer(
        TCHAR_TO_ANSI(*Combined),
        Combined.Len()
    );

    return UnifiedHash.ToString();
}

bool UAIModelArtifactManager::ValidateArtifactPair(const FString& PTPath, const FString& ONNXPath)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    if (!PlatformFile.FileExists(*PTPath) || !PlatformFile.FileExists(*ONNXPath))
    {
        ensureMsgf(false, TEXT("Artifact missing during validation"));
        return false;
    }

    const FString PTHash = ComputeFileHash(PTPath);
    const FString ONNXHash = ComputeFileHash(ONNXPath);

    if (PTHash.IsEmpty() || ONNXHash.IsEmpty())
    {
        ensureMsgf(false, TEXT("Artifact hash failure"));
        return false;
    }

    const FString Unified = ComputeUnifiedArtifactHash(PTPath, ONNXPath);

    return !Unified.IsEmpty();
}

bool UAIModelArtifactManager::IsArtifactPairConsistent(const FModelArtifactSignature& Signature)
{
    const FString RecomputedUnified = ComputeUnifiedArtifactHash(Signature.PTPath, Signature.ONNXPath);

    if (RecomputedUnified.IsEmpty())
    {
        return false;
    }

    return RecomputedUnified == Signature.UnifiedVersionHash;
}

FModelArtifactSignature UAIModelArtifactManager::CreateSignature(
    const FString& PTPath,
    const FString& ONNXPath)
{
    FModelArtifactSignature Signature;

    Signature.PTPath = PTPath;
    Signature.ONNXPath = ONNXPath;

    Signature.PTHash = ComputeFileHash(PTPath);
    Signature.ONNXHash = ComputeFileHash(ONNXPath);

    Signature.UnifiedVersionHash =
        ComputeUnifiedArtifactHash(PTPath, ONNXPath);

    Signature.Timestamp = FDateTime::UtcNow().ToUnixTimestamp();

    return Signature;
}