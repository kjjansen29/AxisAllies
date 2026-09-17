// UAI_CombatSpec.h
#pragma once

// ============================================================
// UAI_CombatSpec.h
// COMBAT CONSTANTS — AXIS & ALLIES 1940 GLOBAL 2E
// ============================================================
// Included only by UAIManager.cpp and translation units that
// perform combat dice sampling or battle state encoding.
// NOT included by AIInferenceSpec.h.
//
// GLOBAL_TECH_OFFSET, TECHS_PER_PLAYER, NUM_TECHNOLOGIES are
// defined in AIInferenceSpec.h and must not be redefined here.
// ============================================================

// ------------------------------------------------------------
// UNIT TYPE INDICES
// Must match FUnitEntity::UnitType and node feature unit count
// block encoding in AIInferenceSpec.h.
// ------------------------------------------------------------
static constexpr int32 UNIT_INFANTRY = 0;
static constexpr int32 UNIT_ARTILLERY = 1;
static constexpr int32 UNIT_MECH_INF = 2;
static constexpr int32 UNIT_TANK = 3;
static constexpr int32 UNIT_FIGHTER = 4;
static constexpr int32 UNIT_TAC_BOMBER = 5;
static constexpr int32 UNIT_STRA_BOMBER = 6;
static constexpr int32 UNIT_AA_GUN = 7;
static constexpr int32 UNIT_SUBMARINE = 8;
static constexpr int32 UNIT_DESTROYER = 9;
static constexpr int32 UNIT_CRUISER = 10;
static constexpr int32 UNIT_BATTLESHIP = 11;
static constexpr int32 UNIT_CARRIER = 12;
static constexpr int32 UNIT_TRANSPORT = 13;

// ------------------------------------------------------------
// BASE ATTACK VALUES
// ------------------------------------------------------------
namespace UnitAttack
{
    static constexpr int32 Infantry = 1;   // 2 when paired with artillery
    static constexpr int32 Artillery = 2;
    static constexpr int32 MechInf = 1;   // 2 with art (or always with Improved Mech)
    static constexpr int32 Tank = 3;
    static constexpr int32 Fighter = 3;   // 4 with Jet Fighters tech
    static constexpr int32 TacBomber = 3;   // 4 when paired with fighter or tank
    static constexpr int32 StraBomber = 4;   // rolls 2 dice best-of with Heavy Bombers
    static constexpr int32 AAGun = 0;
    static constexpr int32 Submarine = 2;   // 3 with Super Submarines tech
    static constexpr int32 Destroyer = 2;
    static constexpr int32 Cruiser = 3;
    static constexpr int32 Battleship = 4;
    static constexpr int32 Carrier = 0;
    static constexpr int32 Transport = 0;
}

// ------------------------------------------------------------
// BASE DEFENSE VALUES
// ------------------------------------------------------------
namespace UnitDefense
{
    static constexpr int32 Infantry = 2;
    static constexpr int32 Artillery = 2;
    static constexpr int32 MechInf = 2;
    static constexpr int32 Tank = 3;
    static constexpr int32 Fighter = 4;
    static constexpr int32 TacBomber = 3;
    static constexpr int32 StraBomber = 1;
    static constexpr int32 AAGun = 0;
    static constexpr int32 Submarine = 1;
    static constexpr int32 Destroyer = 2;
    static constexpr int32 Cruiser = 3;
    static constexpr int32 Battleship = 4;
    static constexpr int32 Carrier = 2;
    static constexpr int32 Transport = 0;
}

// ------------------------------------------------------------
// UNIT PURCHASE COSTS (IPC)
// ------------------------------------------------------------
namespace UnitCost
{
    static constexpr int32 Infantry = 3;
    static constexpr int32 Artillery = 4;
    static constexpr int32 MechInf = 4;
    static constexpr int32 Tank = 6;
    static constexpr int32 Fighter = 10;
    static constexpr int32 TacBomber = 11;
    static constexpr int32 StraBomber = 12;
    static constexpr int32 AAGun = 5;
    static constexpr int32 Submarine = 6;
    static constexpr int32 Destroyer = 8;
    static constexpr int32 Cruiser = 12;
    static constexpr int32 Battleship = 20;
    static constexpr int32 Carrier = 16;
    static constexpr int32 Transport = 7;
    static constexpr int32 MinorIC = 12;
    static constexpr int32 MajorIC = 30;
    static constexpr int32 AirBase = 15;
    static constexpr int32 NavalBase = 15;
}

namespace UnitCostImprovedShipyards
{
    static constexpr int32 Transport = 6;
    static constexpr int32 Submarine = 5;
    static constexpr int32 Destroyer = 7;
    static constexpr int32 Cruiser = 9;
    static constexpr int32 Battleship = 17;
    static constexpr int32 Carrier = 13;
}

// ------------------------------------------------------------
// UNIT MOVEMENT VALUES
// ------------------------------------------------------------
namespace UnitMove
{
    static constexpr int32 Infantry = 1;
    static constexpr int32 Artillery = 1;
    static constexpr int32 MechInf = 2;
    static constexpr int32 Tank = 2;
    static constexpr int32 Fighter = 4;
    static constexpr int32 TacBomber = 4;
    static constexpr int32 StraBomber = 6;
    static constexpr int32 AAGun = 1;
    static constexpr int32 Submarine = 2;
    static constexpr int32 Destroyer = 2;
    static constexpr int32 Cruiser = 2;
    static constexpr int32 Battleship = 2;
    static constexpr int32 Carrier = 2;
    static constexpr int32 Transport = 2;
}

// ------------------------------------------------------------
// CAPITAL SHIP HIT POINTS
// ------------------------------------------------------------
static constexpr int32 CAPITAL_SHIP_HIT_POINTS = 2;

// ------------------------------------------------------------
// AA GUN FIRE
// In regular combat: automatic, round 0 only.
// Targets up to 3 attacking air units + paratroopers (bIsParatrooper).
// Hits on 1 (base) or 1-2 (with Radar tech).
// Facility AA during SBR: separate, one die per bomber per facility.
// ------------------------------------------------------------
static constexpr int32 AA_GUN_HIT_VALUE_BASE = 1;
static constexpr int32 AA_GUN_HIT_VALUE_RADAR = 2;
static constexpr int32 AA_GUN_MAX_TARGETS = 3;

// ------------------------------------------------------------
// SUBMARINE SURPRISE STRIKE
// ------------------------------------------------------------
static constexpr int32 SUB_SURPRISE_ATTACK_HIT = 2;  // 3 with Super Subs
static constexpr int32 SUB_SURPRISE_DEFEND_HIT = 1;

// ------------------------------------------------------------
// BOMBARDMENT
// ------------------------------------------------------------
static constexpr int32 BOMBARD_BATTLESHIP_HIT = 4;
static constexpr int32 BOMBARD_CRUISER_HIT = 3;

// ------------------------------------------------------------
// KAMIKAZE STRIKES
// ------------------------------------------------------------
static constexpr int32 KAMIKAZE_TOTAL = 6;
static constexpr int32 KAMIKAZE_HIT_VALUE = 2;

// ------------------------------------------------------------
// STRATEGIC/TACTICAL BOMBING RAID
// Facility AA: one die per bomber per facility, hit on 1 (or 1-2 with Radar).
// Note: AA gun units in territory do NOT fire during SBR.
// StrategicBombingDecision action encoding:
//   0 = Pass (no SBR)
//   1 = Target IC (strategic bombers only)
//   2 = Target air base
//   3 = Target naval base
//   4-6 = always illegal
// ------------------------------------------------------------
static constexpr int32 FACILITY_AA_HIT_BASE = 1;
static constexpr int32 FACILITY_AA_HIT_RADAR = 2;
static constexpr int32 STRA_BOMBER_DAMAGE_BONUS = 2;
static constexpr int32 MAJOR_IC_DAMAGE_CAP = 20;
static constexpr int32 MINOR_IC_DAMAGE_CAP = 6;
static constexpr int32 BASE_DAMAGE_CAP = 6;
static constexpr int32 MAJOR_IC_BASE_PRODUCTION = 10;
static constexpr int32 MINOR_IC_BASE_PRODUCTION = 3;
static constexpr int32 MAJOR_IC_IFP_PRODUCTION = 12;
static constexpr int32 MINOR_IC_IFP_PRODUCTION = 4;

// SBR air battle: all units attack and defend at 1, one round only.
static constexpr int32 SBR_AIR_BATTLE_ATTACK_VALUE = 1;
static constexpr int32 SBR_AIR_BATTLE_DEFENSE_VALUE = 1;

// ------------------------------------------------------------
// TECHNOLOGY INDEX ORDER
// Access: GlobalFeatures[GLOBAL_TECH_OFFSET + P * TECHS_PER_PLAYER + T]
// GLOBAL_TECH_OFFSET and TECHS_PER_PLAYER defined in AIInferenceSpec.h.
// ------------------------------------------------------------
static constexpr int32 GLOBAL_TECH_LONG_RANGE_AIR = 0;
static constexpr int32 GLOBAL_TECH_IMP_SHIPYARDS = 1;
static constexpr int32 GLOBAL_TECH_RADAR = 2;
static constexpr int32 GLOBAL_TECH_ROCKETS = 3;
static constexpr int32 GLOBAL_TECH_IMPROVED_MECH = 4;
static constexpr int32 GLOBAL_TECH_ADV_ARTILLERY = 5;
static constexpr int32 GLOBAL_TECH_SUPER_SUBS = 6;
static constexpr int32 GLOBAL_TECH_JET_FIGHTERS = 7;
static constexpr int32 GLOBAL_TECH_WAR_BONDS = 8;
static constexpr int32 GLOBAL_TECH_COMBINED_BOMB = 9;
static constexpr int32 GLOBAL_TECH_PARATROOPERS = 10;
static constexpr int32 GLOBAL_TECH_HEAVY_BOMBERS = 11;

// GLOBAL_BATTLE_* offset constants and GLOBAL_KAMIKAZE_REMAINING are
// defined in AIInferenceSpec.h and must not be redefined here.