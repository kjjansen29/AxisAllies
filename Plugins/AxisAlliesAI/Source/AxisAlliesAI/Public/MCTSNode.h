// MCTSNode.h
#pragma once
#include "CoreMinimal.h"
#include "MCTSNode.generated.h"

// ============================================================
// FUnitEntity
// ============================================================
// COUNT:
//   Count = units 100% identical across every field.
//   Units differing in any field are separate entries with Count=1.
//
// SLOT ID:
//   Transports and carriers have unique SlotId (1+) per territory.
//   Non-container units: SlotId = 0.
//   Air units on carriers: SlotId = carrier's SlotId.
//
// CARGO ENCODING:
//   (CargoUnitTypeA + 1) / 14; 0 if empty. Identical for transports and carriers.
//   Transports carry ground units. Carriers carry air units.
//   Air units on carriers also exist as top-level entities with SlotId = carrier's SlotId.
//   Loaded ground units on transports have NO top-level entity entry.
//
// COMMITMENT FLAGS (present-tense):
//   bIsStrategicBombing      — bomber committed to SBR this turn
//   bIsEscorting             — fighter committed as SBR escort this turn
//   bIsIntercepting          — fighter committed as SBR interceptor this turn
//   bIsConductingSurpriseStrike — submarine currently firing surprise strike
//   IsBombarding             — 0=not bombarding; >0 = destination territory index + 1; encoded as value/329.0f
//
// COMPLETION FLAGS (persist full turn, prevent repeat):
//   bHasCompletedSurpriseStrike — sub has already fired surprise strike
//   bHasCompletedBombardment    — ship has already bombarded
//
// BATTLE STATE EXCLUSION:
//   Blueprint excludes units with any commitment or completion flag
//   when building the battle state unit counts for regular combat.
//   C++ SampleCombatDice reads unit counts from the battle state block
//   directly and does not re-filter by entity flags.
// ============================================================
USTRUCT(BlueprintType)
struct FUnitEntity
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite) int32 UnitType = 0;
    UPROPERTY(BlueprintReadWrite) int32 OwningPlayer = 0;
    UPROPERTY(BlueprintReadWrite) int32 Count = 1;
    UPROPERTY(BlueprintReadWrite) int32 HitPoints = 1;
    UPROPERTY(BlueprintReadWrite) int32 MovementRemaining = 0;
    UPROPERTY(BlueprintReadWrite) int32 SlotId = 0;

    UPROPERTY(BlueprintReadWrite) bool bIsScrambled = false;
    UPROPERTY(BlueprintReadWrite) bool bHasLoadedThisTurn = false;
    UPROPERTY(BlueprintReadWrite) bool bHasUnloadedThisTurn = false;
    UPROPERTY(BlueprintReadWrite) bool bIsSubmerged = false;
    UPROPERTY(BlueprintReadWrite) int32 CombatEngagementState = 0;  // if !bIsStrategicBombing: 0=default, 1=has engaged in combat; if bIsStrategicBombing: 0=IC, 1=air base, 2=naval base
    UPROPERTY(BlueprintReadWrite) bool bIsRetreating = false;

    UPROPERTY(BlueprintReadWrite) int32 CargoUnitTypeA = -1;
    UPROPERTY(BlueprintReadWrite) int32 CargoUnitOwnerA = -1;
    UPROPERTY(BlueprintReadWrite) int32 CargoUnitTypeB = -1;
    UPROPERTY(BlueprintReadWrite) int32 CargoUnitOwnerB = -1;

    // Commitment flags
    UPROPERTY(BlueprintReadWrite) bool bIsStrategicBombing = false;
    UPROPERTY(BlueprintReadWrite) bool bIsEscorting = false;
    UPROPERTY(BlueprintReadWrite) bool bIsIntercepting = false;
    UPROPERTY(BlueprintReadWrite) bool bIsConductingSurpriseStrike = false;
    UPROPERTY(BlueprintReadWrite) int32 IsBombarding = 0;  // 0=not bombarding; >0 = destination territory index + 1

    // Completion flags
    UPROPERTY(BlueprintReadWrite) bool bHasCompletedSurpriseStrike = false;
    UPROPERTY(BlueprintReadWrite) bool bHasCompletedBombardment = false;

    // Paratrooper flag
    UPROPERTY(BlueprintReadWrite) bool bIsParatrooper = false;

    // Origin territory at start of turn
    UPROPERTY(BlueprintReadWrite) int32 StartOfTurnTerritory = 0;

    void ToModelFeatures(float* OutFeatures) const
    {
        OutFeatures[0] = UnitType / 13.0f;
        OutFeatures[1] = (OwningPlayer + 1) / 12.0f;
        OutFeatures[2] = FMath::Min(Count, 20) / 20.0f;
        OutFeatures[3] = HitPoints / 2.0f;
        OutFeatures[4] = FMath::Min(MovementRemaining, 8) / 8.0f;
        OutFeatures[5] = SlotId / 20.0f;
        OutFeatures[6] = bIsScrambled ? 1.0f : 0.0f;
        OutFeatures[7] = bHasLoadedThisTurn ? 1.0f : 0.0f;
        OutFeatures[8] = bHasUnloadedThisTurn ? 1.0f : 0.0f;
        OutFeatures[9] = bIsSubmerged ? 1.0f : 0.0f;
        OutFeatures[10] = (float)CombatEngagementState / 2.0f;
        OutFeatures[11] = bIsRetreating ? 1.0f : 0.0f;
        OutFeatures[12] = CargoUnitTypeA >= 0 ? (CargoUnitTypeA + 1) / 14.0f : 0.0f;
        OutFeatures[13] = CargoUnitOwnerA >= 0 ? (CargoUnitOwnerA + 1) / 12.0f : 0.0f;
        OutFeatures[14] = CargoUnitTypeB >= 0 ? (CargoUnitTypeB + 1) / 14.0f : 0.0f;
        OutFeatures[15] = CargoUnitOwnerB >= 0 ? (CargoUnitOwnerB + 1) / 12.0f : 0.0f;
        OutFeatures[16] = bIsStrategicBombing ? 1.0f : 0.0f;
        OutFeatures[17] = bIsEscorting ? 1.0f : 0.0f;
        OutFeatures[18] = bIsIntercepting ? 1.0f : 0.0f;
        OutFeatures[19] = bIsConductingSurpriseStrike ? 1.0f : 0.0f;
        OutFeatures[20] = (float)IsBombarding / 329.0f;
        OutFeatures[21] = bHasCompletedSurpriseStrike ? 1.0f : 0.0f;
        OutFeatures[22] = bHasCompletedBombardment ? 1.0f : 0.0f;
        OutFeatures[23] = bIsParatrooper ? 1.0f : 0.0f;
        OutFeatures[24] = (float)StartOfTurnTerritory / 328.0f;
    }
};

// ============================================================
// FTerritoryEntityList
// ============================================================
USTRUCT(BlueprintType)
struct FTerritoryEntityList
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite)
    TArray<FUnitEntity> Entities;

    void Reset() { Entities.Empty(); }
};

// ============================================================
// FPendingActionContext
// ============================================================
USTRUCT()
struct FPendingActionContext
{
    GENERATED_BODY()

    int32 PendingActionA = INDEX_NONE;
    int32 PendingActionB = INDEX_NONE;  // LoadUnitsCombat / LoadUnitsNonCombat
    int32 PendingActionC = INDEX_NONE;  // CombatMoveUnitDest / NonCombatUnitDest
    int32 PendingActionD = INDEX_NONE;  // Second LoadUnitsCombat / LoadUnitsNonCombat (if transport loads two units)
    bool  bIsValid = false;
};

// ============================================================
// FMCTSNode
// ============================================================
USTRUCT(BlueprintType)
struct FMCTSNode
{
    GENERATED_BODY()

    // Graph transformer input state
    TArray<float> NodeFeatures;    // 329 x 19 = 6,251 floats
    TArray<float> GlobalFeatures;  // 548 floats

    UPROPERTY(BlueprintReadWrite)
    TArray<FTerritoryEntityList> EntityLists;

    FPendingActionContext PendingActionContext;

    // Tree search core
    int32 PhaseId = 0;
    int32 PlayerId = 0;
    int32 VisitCount = 0;
    float TotalValue = 0.0f;

    TArray<float> PolicyPrior;
    TArray<float> MCTSValuePerPlayer;

    // Tree structure
    int32         ActionFromParent = -1;
    int32         ParentIndex = INDEX_NONE;
    TArray<int32> ChildIndices;
    TArray<bool>  LegalActionMask;
    TArray<int32> EdgeVisitCount;
    TArray<float> EdgeValueSum;
    TArray<int32> VirtualLossEdgeCount;
    int32         VirtualLossCount = 0;

    // Metadata
    int32 GameId = 0;
    int32 MoveIndex = 0;

    // Flags
    bool bIsExpanded = false;
    bool bPolicyInitialized = false;
    bool bIsTerminal = false;

    void InitializeEntityLists()
    {
        EntityLists.SetNum(329);
        for (FTerritoryEntityList& List : EntityLists)
            List.Reset();
    }
};

// ============================================================
// FMCTSTree
// ============================================================
USTRUCT(BlueprintType)
struct FMCTSTree
{
    GENERATED_BODY()

    TArray<FMCTSNode> Nodes;
    int32 RootIndex = INDEX_NONE;

    int32 CreateNode(const FMCTSNode& Node)
    {
        return Nodes.Add(Node);
    }

    FMCTSNode& GetNode(int32 Index)
    {
        return Nodes[Index];
    }

    int32 AddChild(int32 ParentIndex, const FMCTSNode& ChildNode)
    {
        FMCTSNode NewChild = ChildNode;
        NewChild.ParentIndex = ParentIndex;
        const int32 ChildIndex = Nodes.Add(NewChild);
        Nodes[ParentIndex].ChildIndices.Add(ChildIndex);
        return ChildIndex;
    }
};