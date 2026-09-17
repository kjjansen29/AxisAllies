# Axis & Allies 1940 AI — Blueprint Function Responsibilities

## Overview

Blueprint handles all game state management, legal action masking, state transitions, actual dice rolling, and entity list maintenance. C++ handles MCTS tree search, GPU inference, training pipeline, and combat dice simulation during MCTS.

The core Blueprint call loop:

```
1. On game start:
       LoadInitialGameState(OutStateBuffer, OutEntityLists, OutPhaseId, OutActingPlayer, OutRound)

2. Each time the AI must act:
       Action = GetActionFromMCTS(StateBuffer, PhaseId, PlayerId, GameStateObject, EntityLists)

3. Apply Action to the live game state.
   Update StateBuffer and EntityLists to reflect the result.

4. Advance to the next phase/player per SimulateTransition's OutPhaseId and OutPlayerId.

5. On game end:
       ShouldTrain = EndAIArenaGame(OutcomeValues)
       if ShouldTrain: FinalizeMCTSTrainingPipeline()
```

---

## Game Start

### LoadInitialGameState
```
bool LoadInitialGameState(
    TArray<float>& OutStateBuffer,       // 6,799 floats
    TArray<FTerritoryEntityList>& OutEntityLists,
    int32& OutPhaseId,
    int32& OutActingPlayer,
    int32& OutRound)
```

Call once at game start. Returns the flat state buffer, entity lists for all 329 territories, and initial phase/player/round.

**State buffer layout:**
- `[0..6250]` = node features (329 × 19)
- `[6251..6798]` = global features (548)

**Initialization requirements:**
- `GlobalFeatures[512]` = 1.0 (kamikaze_remaining = 6/6 for Japan)
- `GlobalFeatures[513..547]` = all 0.0 (no active battle)
- All entity flags default to false

---

## Primary AI Entry Point

### GetActionFromMCTS
```
int32 GetActionFromMCTS(
    const TArray<float>& StateBuffer,
    int32 PhaseId,
    int32 PlayerId,
    UObject* GameStateObject,
    const TArray<FTerritoryEntityList>& EntityLists)
```

C++ runs the full MCTS search and returns the selected action index. Blueprint must provide an accurate StateBuffer and EntityLists at the time of the call.

---

## Blueprint-Implemented Events

### SimulateTransition
```
FApplyActionResult SimulateTransition(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    int32 Action,
    bool bHasPendingContext,
    int32 PendingActionA,
    int32 PendingActionB,
    const TArray<FTerritoryEntityList>& InEntityLists)
```

Called by C++ during MCTS tree expansion for chain-final, standalone, and conditionally-routed phases. Returns: `OutState`, `OutPhaseId`, `OutPlayerId`, `OutLegalActionMask`, `bIsTerminal`, `OutEntityLists`.

`OutEntityLists` must always be populated — copy InEntityLists through even when no entity changes occur.

C++ manages PendingActionContext internally. Blueprint receives PendingActionA/B for its own use and does not need to store or forward them.

#### Chain-final phases (bHasPendingContext = true)

| Phase | PendingActionA | PendingActionB |
|---|---|---|
| TechChartSelection (1) | Dice count purchased (1–6) | INDEX_NONE |
| RepairQuantity (3) | Repair territory (0–328) | INDEX_NONE |
| PurchaseQuantity (5) | Purchase type action (0–18) | INDEX_NONE |
| KamikazeTarget (8) | Quantity (1–6) | KamikazeLocation action (0–5) |
| ScrambleQuantity (11) | Source territory (0–201) | ScrambleUnitDest action |
| BombardQuantity (14) | Source sea zone (0–126) | BombardDest action |
| CombatMoveQuantity (19) | Source territory (0–328) | CombatMoveUnitDest action |
| StrategicBombingQuantity (21) | StrategicBombingDecision action (1–3) | INDEX_NONE |
| SubmarineActionQuantity (23) | SubmarineActionType action (0–2) | INDEX_NONE |
| CombatResolveCasualtyQuantity (25) | CasualtyType action (0–13) | INDEX_NONE |
| NonCombatQuantity (30) | Source territory (0–328) | NonCombatUnitDest action |
| PlacementQuantity (32) | PurchaseType action (0–18) | Placement territory (0–328) |
| SBR_AirBattleCasualtyQuantity (35) | SBR_AirBattleCasualtyType action (0–13) | INDEX_NONE |

#### Standalone phases (bHasPendingContext = false)

| Phase | Blueprint responsibility |
|---|---|
| Tech (0) | Apply tech investment. C++ calls SampleTechDice after expansion if Action > 0. |
| DeclareWar (15) | Apply war declaration. Update GlobalFeatures war status [215–250]. Action 9 = Pass. |
| CombatResolveContinueOrRetreat (26) | Apply retreat (action 1–12 = retreat destination) or continue (action 0). |
| SBR_InterceptorCommitment (33) | Commit selected fighter as interceptor. Set bIsIntercepting on entity. Action 21056 = Pass for this player. Iterates per fighter per player per SBR territory. |

#### Conditionally-routed phases

| Phase | Routing logic |
|---|---|
| CombatMoveSource (16) | Transport selected → 17 (LoadUnitsCombat). Non-transport → 18 (CombatMoveUnitDest). Pass (action 329) → end attacker combat move phase, begin Defender Response. |
| LoadUnitsCombat (17) | Transport has movement remaining after load → 18 (CombatMoveUnitDest). No movement → back to 16 (CombatMoveSource). |
| NonCombatSource (27) | Transport selected → 28 (LoadUnitsNonCombat). Non-transport → 29 (NonCombatUnitDest). Pass (action 329) → end non-combat move. |
| LoadUnitsNonCombat (28) | Transport has movement remaining after load → 29 (NonCombatUnitDest). No movement → back to 27 (NonCombatSource). |

`OutLegalActionMask` must be sized to match the returned `OutPhaseId`'s action space.

### GetLegalActionMask
```
FApplyActionResult GetLegalActionMask(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    bool bHasPendingContext,
    int32 PendingActionA,
    int32 PendingActionB,
    const TArray<FTerritoryEntityList>& InEntityLists)
```

Called by C++ during MCTS tree expansion on chain-intermediate phases and on root node creation.

### GetCasualtyAssignmentMask
```
FApplyActionResult GetCasualtyAssignmentMask(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    const TArray<FTerritoryEntityList>& InEntityLists)
```

Called immediately after `SampleCombatDice` or `SampleSubmarineSurpriseDice`. Returns `OutLegalActionMask` of size 14. Blueprint reads from `GlobalFeatures[544]` (ATK_HITS) and `[545]` (DEF_HITS) to determine which side is assigning casualties, then reads unit counts from `GlobalFeatures[516–543]` to determine which unit types are eligible.

---

## Battle State Management

### When to build battle state
Blueprint writes GlobalFeatures[513–547] when a territory's combat becomes the active battle. The block must be cleared to zero when the battle ends or is resolved.

### What to exclude
When populating attacking and defending unit counts [516–543], Blueprint must exclude any unit entity with any of the following flags set to true:
- `bIsStrategicBombing`
- `bIsEscorting`
- `bIsIntercepting`
- `bIsConductingSurpriseStrike`
- `bIsBombarding`
- `bHasCompletedSurpriseStrike`
- `bHasCompletedBombardment`

### Battle state write sequence

**On battle start:**
```
GlobalFeatures[513] = territory_index / 328.0f
GlobalFeatures[514] = (attacker_player + 1) / 12.0f
GlobalFeatures[515] = (defender_player + 1) / 12.0f
GlobalFeatures[516 + unit_type] = eligible_atk_count / 20.0f  // for each of 14 types
GlobalFeatures[530 + unit_type] = eligible_def_count / 20.0f  // for each of 14 types
GlobalFeatures[544] = 0.0f  // ATK_HITS_PENDING
GlobalFeatures[545] = 0.0f  // DEF_HITS_PENDING
GlobalFeatures[546] = 0.0f  // ROUND_NUMBER
GlobalFeatures[547] = 1.0f if amphibious, else 0.0f  // SEABORNE
```

**After dice are rolled (live game):**
```
GlobalFeatures[544] = atk_hits / 14.0f
GlobalFeatures[545] = def_hits / 14.0f
```

**After casualty assignment completes for a round:**
```
GlobalFeatures[544] = 0.0f
GlobalFeatures[545] = 0.0f
GlobalFeatures[546] = (round_number + 1) / 10.0f
// Also decrement surviving unit counts in [516-543] as units are removed
```

**On battle end:**
```
GlobalFeatures[513..547] = all 0.0f
// Update node features for the territory:
//   NodeFeatures[territory * 19 + 1] = new_controller
```

### Kamikaze hits (live game)
After kamikaze dice are rolled, Blueprint writes hits to `GlobalFeatures[545]` (DEF_HITS_PENDING) for the relevant sea zone's battle. Blueprint then:
1. Transitions to `CombatResolveCasualtyType(24)` for the defender to assign casualties
2. Decrements `GlobalFeatures[512]` (kamikaze_remaining) by the number of strikes committed: `GlobalFeatures[512] = remaining_strikes / 6.0f`

### Bombardment hits (live game)
After bombardment dice are rolled, Blueprint writes hits to `GlobalFeatures[544]` (ATK_HITS_PENDING) then transitions to `CombatResolveCasualtyType(24)`.

### SBR facility damage (live game)
After SBR dice are rolled, Blueprint writes damage to `NodeFeatures[territory * 19 + 5]` (IC damage, encoding = damage / 20.0f) or air/naval base damage at `NodeFeatures[territory * 19 + 7]` or `[9]` (encoding = damage / 6.0f). Blueprint must cap damage at facility limits.

---

## Entity List Responsibilities

### Count field rules
Count > 1 only when units are 100% identical across every field including MovementRemaining, all flags, and cargo. Units differing in any field must be separate entries with Count = 1. Remove entries when Count reaches 0.

### Commitment flag lifecycle

**Set when:** unit is committed to a pre-combat event.
**Clear when:** the event resolves and the unit returns to normal status.

| Flag | Set when | Clear when |
|---|---|---|
| `bIsStrategicBombing` | Bomber assigned to SBR via StrategicBombingDecision | After SBR damage resolves |
| `bIsEscorting` | Fighter committed as SBR escort | After SBR air battle; escort retreats |
| `bIsIntercepting` | Fighter committed as interceptor via SBR_InterceptorCommitment | After SBR air battle |
| `bIsConductingSurpriseStrike` | Sub committed via SubmarineActionType action 0 | After sub surprise casualty assignment |
| `bIsBombarding` | Ship committed via BombardSource/BombardDest/BombardQuantity | After bombardment casualty assignment |

**Completion flags** (persist full turn, reset at start of next turn):

| Flag | Set when |
|---|---|
| `bHasCompletedSurpriseStrike` | After `bIsConductingSurpriseStrike` resolves |
| `bHasCompletedBombardment` | After `bIsBombarding` resolves |

### Paratrooper flag
`bIsParatrooper` = true on infantry units moved as paratroopers during combat move phase. These units are subject to AA gun fire in round 0. C++ `SampleCombatDice` reads entity lists to count paratroopers for AA targeting. Reset to false after combat resolves.

### Scramble flag
`bIsScrambled` = true on fighters/tactical bombers that have been scrambled. These units cannot retreat and must return to origin territory after combat (or move 1 space if origin captured). Reset to false at start of next turn.

### Air units on carriers
Maintain BOTH the carrier's `CargoUnitTypeA/B` and `CargoUnitOwnerA/B` fields AND a separate top-level entity entry for the air unit with `SlotId` = carrier's `SlotId`.

On air unit landing: add top-level entity with carrier's SlotId; set carrier cargo fields.
On air unit launching: remove top-level entity; clear carrier cargo fields.

### Loading/unloading rules
**Combat move phase:**
- Transport can load units and move, or move and unload, but CANNOT both load AND unload in the same turn.
- `bHasLoadedThisTurn` and `bHasUnloadedThisTurn` enforce this.

**Non-combat move phase:**
- Transport CAN both load and unload in the same turn.

### MoveUnitDest action decoding
```
// Decode
int32 SlotId       = Action / 329;
int32 Destination  = Action % 329;

// Encode
int32 Action = SlotId * 329 + Destination;

// Pass
int32 Pass = 21056;
```

---

## Scramble Rules

### Eligibility
- Source territory: coastal or island territory with an operative air base (not destroyed, damage < 6)
- Unit types: fighters and tactical bombers only (no strategic bombers)
- Maximum 3 units per air base per sea zone being attacked (split across sea zones is allowed but total from one air base ≤ 3)
- Allied players can scramble their own units if at war with the attacker
- Each defending player decides separately in turn order
- Pass per player independently

### After combat
Surviving scrambled units must return to origin territory during non-combat move phase. If origin captured: move 1 space to friendly territory or friendly carrier. If no landing space: unit is lost.

---

## SBR Rules

### Sequence
1. Attacker moves bombers and escort fighters to target territory during combat move phase. Blueprint sets `bIsStrategicBombing` on bombers, `bIsEscorting` on escorts.
2. Defender commits interceptors via `SBR_InterceptorCommitment(33)`. Blueprint sets `bIsIntercepting` on selected fighters.
3. If interceptors committed: one round of air battle at attack/defense value 1. Both sides take casualties via existing casualty phases. Escorts retreat (clear `bIsEscorting`).
4. Facility AA fire: C++ automatic. 1 die per bomber per facility, hit on 1 (or 1-2 with Radar). AA gun units in territory do NOT participate.
5. Bombing damage: C++ automatic. Each bomber rolls 1 die; strategic bombers add +2.
6. Blueprint writes damage to node features. Sets `bHasCompletedSurpriseStrike` equivalent on bombers via `bIsStrategicBombing` → false after resolution.

### Facility damage caps
- Major IC: max 20 damage. Encoding: damage / 20.0f
- Minor IC: max 6 damage. Encoding: damage / 20.0f (same scale)
- Air base: max 6 damage. Encoding: damage / 6.0f
- Naval base: max 6 damage. Encoding: damage / 6.0f

### After SBR
Bombers that conducted SBR cannot participate in any other combat. They return during non-combat move phase. Blueprint clears `bIsStrategicBombing` after resolution.

---

## National Objectives

GlobalFeatures[156–192]. Blueprint defines slot mapping and writes values. Most are binary (0.0 or 1.0). Slot 11 (global index 167) = objective value / 40 for variable-value objectives.

---

## Victory State

GlobalFeatures[193–205]:
| Index | Feature | Encoding |
|---|---|---|
| [193–201] | Victory city count per player (9) | count / 50 |
| [202] | Total Axis victory cities | count / 50 |
| [203] | Total Allied victory cities | count / 50 |
| [204] | Allied won | binary |
| [205] | Axis won | binary |

---

## War Status

GlobalFeatures[215–250]. Update immediately when war is declared.

```
// Index for pair (PlayerA, PlayerB):
int32 I = min(PlayerA, PlayerB);
int32 J = max(PlayerA, PlayerB);
int32 Index = 215 + (I * (17 - I) / 2) + (J - I - 1);
GlobalFeatures[Index] = 1.0f;  // at war
```

---

## Game End and Training

### EndAIArenaGame
```
bool EndAIArenaGame(const TArray<float>& FinalOutcomeValues)
```
9-element array: +1.0 for winners, -1.0 for losers. Returns true when training threshold is crossed.

### FinalizeMCTSTrainingPipeline
```
void FinalizeMCTSTrainingPipeline()
```
Call when EndAIArenaGame returns true.

---

## What Blueprint Never Needs To Do

- Manage the MCTS tree, inference queue, or replay buffer
- Store or forward PendingActionA/B — C++ manages PendingActionContext
- Roll dice for MCTS simulation — C++ handles this internally via Sample* functions
- Call ResetMCTS — automatic
- Know which NNE runtime is in use
- Manage model loading beyond EnsureModelsExist
- Write to GlobalFeatures[506] (pending sub-action value) — C++ writes this automatically on chain-final nodes
