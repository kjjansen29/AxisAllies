#pragma once
#include "CoreMinimal.h"


// ----------------------------------------------------------------
// Path helpers — work correctly in both editor and packaged builds
// ----------------------------------------------------------------
FORCEINLINE FString GetPythonAIDir()
{
    return FPaths::ConvertRelativePathToFull(
        FPaths::Combine(FPaths::ProjectContentDir(), TEXT("PythonAI")));
}

FORCEINLINE FString GetAIDataDir()
{
    return FPaths::ConvertRelativePathToFull(
        FPaths::Combine(FPaths::ProjectContentDir(), TEXT("AIData")));
}

FORCEINLINE FString GetTotalReplayBufferExportPath()
{
    return FPaths::Combine(
        FPaths::ProjectSavedDir(),
        TEXT("AITraining/Replay/index.json"));
}

FORCEINLINE FString GetONNXPath()
{
    return FPaths::Combine(
        GetPythonAIDir(),
        TEXT("axis_allies.onnx"));
}
FORCEINLINE FString GetPTPath()
{
    return FPaths::Combine(
        GetPythonAIDir(),
        TEXT("axis_allies.pt"));
}
FORCEINLINE FString GetTrainingMetadataPath()
{
    return FPaths::Combine(
        FPaths::ProjectSavedDir(),
        TEXT("AITraining/TrainingMetadata.json"));
}

// ============================================================
// AIInferenceSpec.h
// GRAPH TRANSFORMER STATE CONTRACT — AXIS & ALLIES 1940 GLOBAL 2E
// ============================================================

static constexpr int32 NUM_TERRITORIES = 329;

// 19 features per territory node.
// [0]  IPC value                          raw IPC / 20
// [1]  Current controller                 (player + 1) / 12
// [2]  Original controller                (player + 1) / 12
// [3]  Minor IC present                   binary
// [4]  Major IC present                   binary
// [5]  IC damage                          damage / 20
// [6]  Air base present                   binary
// [7]  Air base damage                    damage / 6
// [8]  Naval base present                 binary
// [9]  Naval base damage                  damage / 6
// [10] Is sea zone                        binary (static)
// [11] Is victory city                    binary (static)
// [12] Is capital territory               binary (static)
// [13] Is canal-controlling territory     binary (static)
// [14] Is convoy zone                     binary (static)
// [15] Is kamikaze zone                   binary (static)
// [16] Scramble capable                   binary (dynamic)
// [17] Is island                          binary (static)
// [18] Is coastal                         binary (static)
static constexpr int32 NODE_FEATURE_COUNT = 20;

static constexpr int32 NODE_IS_ISLAND = 17;
static constexpr int32 NODE_IS_COASTAL = 18;

// 24 features per unit entity.
// [0]  Unit type              unit_type_index / 13
// [1]  Owning player          (player_index + 1) / 12
// [2]  Count                  min(count, 20) / 20
// [3]  Hit points             hp / 2
// [4]  Movement remaining     min(movement, 8) / 8
// [5]  Slot ID                slot_id / 20   (max slot ID = 19, range 0-19)
// [6]  Is scrambled           binary
// [7]  Has loaded this turn   binary
// [8]  Has unloaded this turn binary
// [9]  Is submerged           binary
// [10] Has blitzed            binary
// [11] Is retreating          binary
// [12] Cargo unit type A      (CargoUnitTypeA + 1) / 14; 0 if empty
// [13] Cargo unit owner A     (CargoUnitOwnerA + 1) / 12; 0 if empty
// [14] Cargo unit type B      (CargoUnitTypeB + 1) / 14; 0 if empty
// [15] Cargo unit owner B     (CargoUnitOwnerB + 1) / 12; 0 if empty
// [16] Is strategic bombing   binary
// [17] Is escorting           binary
// [18] Is intercepting        binary
// [19] Is conducting surprise strike  binary
// [20] Is bombarding          binary
// [21] Has completed surprise strike  binary
// [22] Has completed bombardment      binary
// [23] Is paratrooper         binary
//
// SLOT ID NOTE:
//   Slot IDs range 0-19 (max 20 container units per territory).
//   Normalization: slot_id / 20.0f.
//   Action spaces encode slot IDs directly as integers in the
//   action index formula: slot_id * 329 + territory_index.
//   Maximum action index (slot_id=19, territory=328): 19*329+328 = 6,579.
//   Plus Pass action = 6,580. Total action space = 6,581 (Head6581).
static constexpr int32 UNIT_ENTITY_FEATURE_COUNT = 25;
static constexpr int32 MAX_UNIT_ENTITIES_PER_NODE = 20;
static constexpr int32 ENTITY_HIDDEN_DIM = 64;
static constexpr int32 MAX_SLOT_ID = 19;
static constexpr int32 MAX_CONTAINER_UNITS = 20;

// 593 global features.
// [0-511]   Existing board-wide context (unchanged)
// [512]     kamikaze_remaining          count / 6; initial = 1.0 for Japan
// [513]     Battle territory ID         territory_index / 328
// [514]     Battle attacker             (player + 1) / 12
// [515]     Battle defender             (player + 1) / 12
// [516-529] Attacking unit counts       count[unit_type] / 20
// [530-543] Defending unit counts       count[unit_type] / 20
// [544]     ATK_HITS_PENDING            hits / 14
// [545]     DEF_HITS_PENDING            hits / 14
// [546]     ROUND_NUMBER                round / 10
// [547]     SEABORNE                    binary
static constexpr int32 GLOBAL_FEATURE_COUNT = 593;

static constexpr int32 GLOBAL_KAMIKAZE_REMAINING = 557;
static constexpr int32 GLOBAL_BATTLE_TERRITORY_ID = 558;
static constexpr int32 GLOBAL_BATTLE_ATTACKER = 559;
static constexpr int32 GLOBAL_BATTLE_DEFENDER = 560;
static constexpr int32 GLOBAL_BATTLE_ATK_UNITS = 561;
static constexpr int32 GLOBAL_BATTLE_DEF_UNITS = 575;
static constexpr int32 GLOBAL_BATTLE_ATK_HITS = 589;
static constexpr int32 GLOBAL_BATTLE_DEF_HITS = 590;
static constexpr int32 GLOBAL_BATTLE_ROUND_NUMBER = 591;
static constexpr int32 GLOBAL_BATTLE_SEABORNE = 592;
static constexpr int32 GLOBAL_OUTCOME_VALUES = 418;  // 9 floats [418-426]; +1.0f=win, -1.0f=loss, 0.0f=draw; written by Blueprint on terminal nodes

static constexpr int32 NUM_PLAYERS = 9;
static constexpr int32 NUM_UNIT_TYPES = 14;
static constexpr int32 NUM_TECHNOLOGIES = 12;
static constexpr int32 NUM_NAT_OBJ = 37;
static constexpr int32 MAX_INFERENCE_BATCH_SIZE = 512;
// Fixed batch size of the exported ONNX model. Must match EXPORT_BATCH_SIZE in model.py.
static constexpr int32 INFERENCE_BATCH_SIZE = 32;

static constexpr int32 GLOBAL_TECH_OFFSET = 39;
static constexpr int32 TECHS_PER_PLAYER = 12;

// ------------------------------------------------------------
// STATIC TERRITORY CLASSIFICATION
// ------------------------------------------------------------
// Island territory indices (37 total):
//   7(West Indies), 44(Sardinia), 45(Sicily), 46(Malta), 59(Crete),
//   60(Cyprus), 126(French Madagascar), 131(Ceylon), 135(Sumatra),
//   136(Borneo), 137(Java), 138(Celebes), 139(Dutch New Guinea),
//   140(New Guinea), 159(Hainan), 160(Formosa), 161(Okinawa),
//   171(Iwo Jima), 172(Marianas), 173(Guam), 174(Paulau Island),
//   175(Caroline Islands), 176(Marshall Islands), 177(Wake Island),
//   178(Midway), 179(Johnston Island), 180(Hawaiian Islands),
//   181(Gilbert Islands), 182(Solomon Islands), 183(New Britain),
//   184(New Hebrides), 185(Fiji), 186(New Zealand), 196(Aleutian Islands),
//   199(Samoa), 200(Line Islands), 201(Philippines)
//
// Inland (non-coastal) territory indices (45 total):
//   22(Bolivia), 23(Paraguay), 35(France), 41(Switzerland),
//   49(Greater Southern Germany), 51(Eastern Poland), 52(Slovakia Hungary),
//   67(Vologda), 69(Belarus), 70(Smolensk), 71(Russia), 72(Bryansk),
//   73(Tambov), 74(Samara), 75(Volgograd), 76(Western Ukraine),
//   80(Urals), 82(Novosibirsk), 83(Timguska), 84(Yenisey),
//   85(Yakut S.S.R.), 86(Buryatia), 89(Sakha), 99(Afghanistan),
//   116(French Central Africa), 123(Rhodesia), 141(Tsinghai), 142(Sikang),
//   143(Kansu), 144(Shensi), 145(Szechwan), 146(Kweichow), 147(Yunnan),
//   148(Hunan), 151(Hopei), 152(Anhwe), 154(Suiyuyan), 155(Chahar),
//   162(Olgiy), 163(Tsagaan-Olom), 164(Dzavhan), 165(Central Mongolia),
//   166(Ulaanbaatar), 167(Buyant-Uhaa), 195(Evenkiyskiy)
//
// All remaining land territories (120) are coastal.

FORCEINLINE bool IsIslandTerritory(int32 TerritoryIndex)
{
    switch (TerritoryIndex)
    {
    case 7: case 44: case 45: case 46: case 59: case 60:
    case 126: case 131: case 135: case 136: case 137: case 138:
    case 139: case 140: case 159: case 160: case 161:
    case 171: case 172: case 173: case 174: case 175: case 176:
    case 177: case 178: case 179: case 180: case 181: case 182:
    case 183: case 184: case 185: case 186: case 196:
    case 199: case 200: case 201:
        return true;
    default:
        return false;
    }
}

FORCEINLINE bool IsInlandTerritory(int32 TerritoryIndex)
{
    switch (TerritoryIndex)
    {
    case 22: case 23: case 35: case 41: case 49: case 51: case 52:
    case 67: case 69: case 70: case 71: case 72: case 73: case 74:
    case 75: case 76: case 80: case 82: case 83: case 84: case 85:
    case 86: case 89: case 99: case 116: case 123:
    case 141: case 142: case 143: case 144: case 145: case 146:
    case 147: case 148: case 151: case 152: case 154: case 155:
    case 162: case 163: case 164: case 165: case 166: case 167:
    case 195:
        return true;
    default:
        return false;
    }
}

FORCEINLINE bool IsCoastalTerritory(int32 TerritoryIndex)
{
    if (TerritoryIndex >= 202) return false;
    if (IsIslandTerritory(TerritoryIndex)) return false;
    if (IsInlandTerritory(TerritoryIndex)) return false;
    return true;
}

// ------------------------------------------------------------
// VALIDATION
// ------------------------------------------------------------
static FORCEINLINE bool ValidateGraphTransformerContract(
    int32 NodeCount,
    int32 NodeFeatureCount,
    int32 GlobalFeatureCount,
    int32 UnitEntityFeatureCount)
{
    return NodeCount == NUM_TERRITORIES &&
        NodeFeatureCount == NODE_FEATURE_COUNT &&
        GlobalFeatureCount == GLOBAL_FEATURE_COUNT &&
        UnitEntityFeatureCount == UNIT_ENTITY_FEATURE_COUNT;
}

// ------------------------------------------------------------
// PHASE ENUM (36 phases)
// ------------------------------------------------------------
enum class EPhaseId : int32
{
    Tech = 0,
    TechChartSelection = 1,
    UnitRepair = 2,
    RepairQuantity = 3,
    PurchaseType = 4,
    PurchaseQuantity = 5,
    KamikazeQuantity = 6,
    KamikazeLocation = 7,
    KamikazeTarget = 8,
    ScrambleSource = 9,
    ScrambleUnitDest = 10,
    ScrambleQuantity = 11,
    BombardSource = 12,
    BombardDest = 13,
    BombardQuantity = 14,
    DeclareWar = 15,
    CombatMoveSource = 16,
    LoadUnitsCombat = 17,
    CombatMoveUnitDest = 18,
    CombatMoveQuantity = 19,
    StrategicBombingSource = 20,
    StrategicBombingDecision = 21,
    StrategicBombingQuantity = 22,
    SubmarineActionSource = 23,
    SubmarineActionType = 24,
    SubmarineActionQuantity = 25,
    CombatResolveCasualtyType = 26,
    CombatResolveCasualtyQuantity = 27,
    CombatResolveContinueOrRetreat = 28,
    NonCombatSource = 29,
    LoadUnitsNonCombat = 30,
    NonCombatUnitDest = 31,
    NonCombatQuantity = 32,
    PlacementTerritory = 33,
    PlacementQuantity = 34,
    SBR_InterceptorCommitment = 35,
    SBR_InterceptorQuantity = 36,
    SBR_AirBattleCasualtyType = 37,
    SBR_AirBattleCasualtyQuantity = 38,
    SBR_EscortCommitment = 39,
    SBR_EscortQuantity = 40,
    PlacementCarrier = 41,
    AirUnitLandOnCarrier = 42,
    AirUnitLandOnCarrierDest = 43,
    AirUnitLandOnCarrierPlaneQuantity = 44,
    AirUnitLandOnCarrierCount = 45,
    Count = 46,
};

constexpr int32 PHASE_COUNT = static_cast<int32>(EPhaseId::Count);

// ------------------------------------------------------------
// POLICY HEAD ENUM (17 heads)
// Head2:    TechChartSelection
// Head3:    ScrambleQuantity, SubmarineActionType, StrategicBombingDecision
// Head7:    Tech, KamikazeQuantity
// Head10:   DeclareWar
// Head13:   CombatResolveContinueOrRetreat
// Head14:   CombatResolveCasualtyType, SBR_AirBattleCasualtyType
// Head20:   RepairQuantity, PurchaseQuantity,
//           ScrambleUnitDest, BombardQuantity, CombatMoveQuantity,
//           StrategicBombingQuantity, SubmarineActionQuantity,
//           CombatResolveCasualtyQuantity, NonCombatQuantity,
//           PlacementQuantity, SBR_InterceptorQuantity, SBR_EscortQuantity, SBR_AirBattleCasualtyQuantity
// Head49:   LoadUnitsCombat, LoadUnitsNonCombat
// KamikazeTarget now uses Head20
// Head128:  BombardSource
//ead405)
// Head330:  CombatMoveUnitDest, NonCombatUnitDest, PlacementTerritory
// Head202:  (unused - kept for future use)
// Head6581: CombatMoveSource, NonCombatSource, StrategicBombingSource, SubmarineActionSource, PlacementCarrier, SBR_InterceptorCommitment(SimTrans), SBR_EscortCommitment(SimTrans)
//           Encoding: slot_id * 329 + territory_index; Pass = 6580
//           slot_id range: 0-19 (MAX_SLOT_ID=19, MAX_CONTAINER_UNITS=20)
// Head988: UnitRepair (SimulateTransition)
//           Encoding: repair_type * 329 + territory_index; Pass = 987
//           repair_type: 0=IC, 1=air base, 2=naval base
//           repair_selector: 0=IC, 1=air base, 2=naval base, 3-22=slot_id 0-19
// ------------------------------------------------------------
enum class EPolicyHead : int32
{
    Head2 = 0,
    Head3 = 1,
    Head7 = 3,
    Head10 = 4,
    Head13 = 5,
    Head14 = 6,
    Head20 = 7,
    Head128 = 9,
    Head330 = 10,
    Head6581 = 11,
    Head988 = 12,
    Head49 = 13,
    Head202 = 14,
};

namespace PolicyHeadSize
{
    constexpr int32 Head2 = 2;
    constexpr int32 Head3 = 3;
    constexpr int32 Head7 = 7;
    constexpr int32 Head10 = 10;
    constexpr int32 Head13 = 13;
    constexpr int32 Head14 = 14;
    constexpr int32 Head20 = 20;
    constexpr int32 Head128 = 128;
    constexpr int32 Head49 = 49;
    constexpr int32 Head202 = 202;
    constexpr int32 Head330 = 330;
    constexpr int32 Head6581 = 6581;
    constexpr int32 Head988 = 988;
}

// Encoding helpers
// slot_id * 329 + territory_index; Pass = Head6581 - 1 = 6580
FORCEINLINE int32 EncodeMoveUnitDest(int32 SlotId, int32 TerritoryIndex)
{
    return SlotId * 329 + TerritoryIndex;
}
FORCEINLINE int32 DecodeMoveUnitDestSlot(int32 Action)
{
    return Action / 329;
}
FORCEINLINE int32 DecodeMoveUnitDestTerritory(int32 Action)
{
    return Action % 329;
}

// repair_type * 329 + territory_index; Pass = Head988 - 1 = 987
// repair_type: 0=IC, 1=air base, 2=naval base (ships repair automatically)
// repair_selector: 0=IC, 1=air base, 2=naval base, 3-22=slot_id 0-19
FORCEINLINE int32 EncodeUnitRepair(int32 RepairSelector, int32 TerritoryIndex)
{
    return RepairSelector * 329 + TerritoryIndex;
}
FORCEINLINE int32 DecodeUnitRepairSelector(int32 Action)
{
    return Action / 329;
}
FORCEINLINE int32 DecodeUnitRepairTerritory(int32 Action)
{
    return Action % 329;
}
FORCEINLINE bool UnitRepairIsFacility(int32 RepairSelector)
{
    return RepairSelector <= 2;
}
FORCEINLINE int32 UnitRepairSlotId(int32 RepairSelector)
{
    return RepairSelector - 3;  // valid only when RepairSelector >= 3
}

// ------------------------------------------------------------
// ROUTING FUNCTIONS
// ------------------------------------------------------------
FORCEINLINE EPolicyHead GetPolicyHeadForPhase(EPhaseId Phase)
{
    switch (Phase)
    {
    case EPhaseId::TechChartSelection:
        return EPolicyHead::Head2;

    case EPhaseId::ScrambleQuantity:
        return EPolicyHead::Head3;

    case EPhaseId::SubmarineActionType:
        return EPolicyHead::Head2;

    case EPhaseId::KamikazeLocation:
        return EPolicyHead::Head128;

    case EPhaseId::Tech:
    case EPhaseId::KamikazeQuantity:
        return EPolicyHead::Head7;

    case EPhaseId::StrategicBombingDecision:
        return EPolicyHead::Head3;

    case EPhaseId::StrategicBombingSource:
        return EPolicyHead::Head6581;

    case EPhaseId::DeclareWar:
        return EPolicyHead::Head10;

    case EPhaseId::CombatResolveContinueOrRetreat:
        return EPolicyHead::Head13;

    case EPhaseId::CombatResolveCasualtyType:
    case EPhaseId::SBR_AirBattleCasualtyType:
        return EPolicyHead::Head20;

    case EPhaseId::RepairQuantity:
    case EPhaseId::PurchaseType:
    case EPhaseId::PurchaseQuantity:
    case EPhaseId::BombardQuantity:
    case EPhaseId::CombatMoveQuantity:
    case EPhaseId::StrategicBombingQuantity:
    case EPhaseId::SubmarineActionQuantity:
    case EPhaseId::CombatResolveCasualtyQuantity:
    case EPhaseId::NonCombatQuantity:
    case EPhaseId::PlacementQuantity:
    case EPhaseId::SBR_InterceptorQuantity:
    case EPhaseId::SBR_EscortQuantity:
    case EPhaseId::SBR_AirBattleCasualtyQuantity:
        return EPolicyHead::Head20;

    case EPhaseId::LoadUnitsCombat:
    case EPhaseId::LoadUnitsNonCombat:
        return EPolicyHead::Head49;

    case EPhaseId::KamikazeTarget:
        return EPolicyHead::Head20;

    case EPhaseId::ScrambleUnitDest:
        return EPolicyHead::Head128;

    case EPhaseId::BombardSource:
        return EPolicyHead::Head6581;

    case EPhaseId::ScrambleSource:
        return EPolicyHead::Head6581;

    case EPhaseId::CombatMoveSource:
    case EPhaseId::NonCombatSource:
    case EPhaseId::SBR_EscortCommitment:
        return EPolicyHead::Head6581;

    case EPhaseId::PlacementTerritory:
        return EPolicyHead::Head330;

    case EPhaseId::PlacementCarrier:
        return EPolicyHead::Head6581;

    case EPhaseId::AirUnitLandOnCarrier:
        return EPolicyHead::Head6581;

    case EPhaseId::AirUnitLandOnCarrierDest:
        return EPolicyHead::Head20;

    case EPhaseId::AirUnitLandOnCarrierPlaneQuantity:
    case EPhaseId::AirUnitLandOnCarrierCount:
        return EPolicyHead::Head20;

    case EPhaseId::BombardDest:
        return EPolicyHead::Head202;

    case EPhaseId::CombatMoveUnitDest:
    case EPhaseId::NonCombatUnitDest:
        return EPolicyHead::Head330;

    case EPhaseId::SBR_InterceptorCommitment:
        return EPolicyHead::Head6581;

    case EPhaseId::SubmarineActionSource:
        return EPolicyHead::Head6581;

    case EPhaseId::UnitRepair:
        return EPolicyHead::Head988;

    default:
        ensureMsgf(false, TEXT("Invalid EPhaseId passed to GetPolicyHeadForPhase"));
        return EPolicyHead::Head7;
    }
}

FORCEINLINE int32 GetPolicySizeForHead(EPolicyHead Head)
{
    switch (Head)
    {
    case EPolicyHead::Head2:    return PolicyHeadSize::Head2;
    case EPolicyHead::Head3:    return PolicyHeadSize::Head3;
    case EPolicyHead::Head7:    return PolicyHeadSize::Head7;
    case EPolicyHead::Head10:   return PolicyHeadSize::Head10;
    case EPolicyHead::Head13:   return PolicyHeadSize::Head13;
    case EPolicyHead::Head14:   return PolicyHeadSize::Head14;
    case EPolicyHead::Head20:   return PolicyHeadSize::Head20;
    case EPolicyHead::Head49:   return PolicyHeadSize::Head49;
    case EPolicyHead::Head128:  return PolicyHeadSize::Head128;
    case EPolicyHead::Head202:  return PolicyHeadSize::Head202;
    case EPolicyHead::Head330:  return PolicyHeadSize::Head330;
    case EPolicyHead::Head6581: return PolicyHeadSize::Head6581;
    case EPolicyHead::Head988: return PolicyHeadSize::Head988;
    default:                    return 0;
    }
}

FORCEINLINE int32 GetPolicySizeForPhase(EPhaseId Phase)
{
    return GetPolicySizeForHead(GetPolicyHeadForPhase(Phase));
}

// ------------------------------------------------------------
// CHAIN ROUTING
// ------------------------------------------------------------
FORCEINLINE bool IsChainIntermediatePhase(EPhaseId Phase)
{
    switch (Phase)
    {

    case EPhaseId::KamikazeLocation:
    case EPhaseId::ScrambleUnitDest:
    case EPhaseId::BombardDest:
    case EPhaseId::CombatMoveUnitDest:
    case EPhaseId::NonCombatUnitDest:
    case EPhaseId::StrategicBombingDecision:
    case EPhaseId::SubmarineActionType:
    case EPhaseId::CombatResolveCasualtyType:
    case EPhaseId::PlacementTerritory:
    case EPhaseId::AirUnitLandOnCarrierDest:
    case EPhaseId::AirUnitLandOnCarrierPlaneQuantity:
    case EPhaseId::SBR_AirBattleCasualtyType:
        return true;
    default:
        return false;
    }
}

FORCEINLINE EPhaseId GetNextChainPhase(EPhaseId Phase)
{
    switch (Phase)
    {
    case EPhaseId::KamikazeLocation:          return EPhaseId::KamikazeTarget;
    case EPhaseId::ScrambleUnitDest:          return EPhaseId::ScrambleQuantity;
    case EPhaseId::BombardDest:               return EPhaseId::BombardQuantity;
    case EPhaseId::CombatMoveUnitDest:        return EPhaseId::CombatMoveQuantity;
    case EPhaseId::StrategicBombingDecision:  return EPhaseId::StrategicBombingQuantity;
    case EPhaseId::SubmarineActionType:       return EPhaseId::SubmarineActionQuantity;
    case EPhaseId::CombatResolveCasualtyType: return EPhaseId::CombatResolveCasualtyQuantity;
    case EPhaseId::NonCombatUnitDest:         return EPhaseId::NonCombatQuantity;
    case EPhaseId::PlacementTerritory:        return EPhaseId::PlacementQuantity;
    case EPhaseId::SBR_AirBattleCasualtyType:    return EPhaseId::SBR_AirBattleCasualtyQuantity;
    case EPhaseId::AirUnitLandOnCarrierDest:          return EPhaseId::AirUnitLandOnCarrierPlaneQuantity;
    case EPhaseId::AirUnitLandOnCarrierPlaneQuantity: return EPhaseId::AirUnitLandOnCarrierCount;
    default:
        ensureMsgf(false, TEXT("GetNextChainPhase called on non-intermediate phase"));
        return Phase;
    }
}

FORCEINLINE bool IsSecondStepOfThreeStepChain(EPhaseId Phase)
{
    return Phase == EPhaseId::StrategicBombingDecision ||
        Phase == EPhaseId::KamikazeLocation ||
        Phase == EPhaseId::ScrambleUnitDest ||
        Phase == EPhaseId::BombardDest ||
        Phase == EPhaseId::SubmarineActionType ||
        Phase == EPhaseId::AirUnitLandOnCarrierDest;
}

FORCEINLINE bool IsThirdStepOfFourStepChain(EPhaseId Phase)
{
    return Phase == EPhaseId::CombatMoveUnitDest ||
        Phase == EPhaseId::NonCombatUnitDest ||
        Phase == EPhaseId::AirUnitLandOnCarrierPlaneQuantity;
}

FORCEINLINE bool IsChainFinalPhase(EPhaseId Phase)
{
    switch (Phase)
    {
    case EPhaseId::TechChartSelection:
    case EPhaseId::RepairQuantity:
    case EPhaseId::PurchaseQuantity:
    case EPhaseId::KamikazeTarget:
    case EPhaseId::ScrambleQuantity:
    case EPhaseId::BombardQuantity:
    case EPhaseId::CombatMoveQuantity:
    case EPhaseId::StrategicBombingQuantity:
    case EPhaseId::SubmarineActionQuantity:
    case EPhaseId::CombatResolveCasualtyQuantity:
    case EPhaseId::NonCombatQuantity:
    case EPhaseId::PlacementQuantity:
    case EPhaseId::SBR_InterceptorQuantity:
    case EPhaseId::SBR_EscortQuantity:
    case EPhaseId::SBR_AirBattleCasualtyQuantity:
    case EPhaseId::AirUnitLandOnCarrierCount:
        return true;
    default:
        return false;
    }
}

FORCEINLINE int32 GetPrecedingChainPhaseActionSize(EPhaseId ChainFinalPhase)
{
    switch (ChainFinalPhase)
    {
    case EPhaseId::TechChartSelection:            return PolicyHeadSize::Head7;
    case EPhaseId::RepairQuantity:                return PolicyHeadSize::Head988;
    case EPhaseId::PurchaseQuantity:              return PolicyHeadSize::Head20;
    case EPhaseId::KamikazeTarget:                return PolicyHeadSize::Head128;
    case EPhaseId::ScrambleQuantity:              return PolicyHeadSize::Head128;
    case EPhaseId::BombardQuantity:               return PolicyHeadSize::Head202;
    case EPhaseId::CombatMoveQuantity:            return PolicyHeadSize::Head330;
    case EPhaseId::StrategicBombingQuantity:      return PolicyHeadSize::Head3;
    case EPhaseId::SubmarineActionQuantity:       return PolicyHeadSize::Head2;
    case EPhaseId::CombatResolveCasualtyQuantity: return PolicyHeadSize::Head20;
    case EPhaseId::NonCombatQuantity:             return PolicyHeadSize::Head330;
    case EPhaseId::PlacementQuantity:             return 0;  // GlobalFeatures[551] not used; preceding phase determined by Blueprint routing
    case EPhaseId::SBR_InterceptorQuantity:        return PolicyHeadSize::Head6581;
    case EPhaseId::SBR_EscortQuantity:              return PolicyHeadSize::Head6581;
    case EPhaseId::SBR_AirBattleCasualtyQuantity:   return PolicyHeadSize::Head20;
    case EPhaseId::AirUnitLandOnCarrierCount:           return PolicyHeadSize::Head20;
    default:
        ensureMsgf(false,
            TEXT("GetPrecedingChainPhaseActionSize: non-chain-final phase %d"),
            static_cast<int32>(ChainFinalPhase));
        return 1;
    }
}