#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/SecureHash.h"
#include "UAIModelArtifactManager.generated.h"

USTRUCT(BlueprintType)
struct FModelArtifactSignature
{
    GENERATED_BODY()

    UPROPERTY()
    FString PTPath;

    UPROPERTY()
    FString ONNXPath;

    UPROPERTY()
    FString PTHash;

    UPROPERTY()
    FString ONNXHash;

    UPROPERTY()
    FString UnifiedVersionHash;

    UPROPERTY()
    int64 Timestamp = 0;
};

UCLASS()
class AXISALLIESAI_API UAIModelArtifactManager : public UObject
{
    GENERATED_BODY()

public:

    // ------------------------------------------------------------
    // HASHING
    // ------------------------------------------------------------
    static FString ComputeFileHash(const FString& FilePath);

    static FString ComputeUnifiedArtifactHash(const FString& PTPath, const FString& ONNXPath);

    // ------------------------------------------------------------
    // VALIDATION
    // ------------------------------------------------------------
    static bool ValidateArtifactPair(const FString& PTPath, const FString& ONNXPath);

    static bool IsArtifactPairConsistent(const FModelArtifactSignature& Signature);

    // ------------------------------------------------------------
    // VERSION BINDING
    // ------------------------------------------------------------
    static FModelArtifactSignature CreateSignature(const FString& PTPath, const FString& ONNXPath);

    static bool IsNewerSignature(const FModelArtifactSignature& NewSig, const FModelArtifactSignature& OldSig);

    // ------------------------------------------------------------
    // SAFETY GATE (USED BY RDG SWAP)
    // ------------------------------------------------------------
    static bool ApproveModelSwap(const FString& PTPath, const FString& ONNXPath, const FString& ExpectedReplayVersion);
};