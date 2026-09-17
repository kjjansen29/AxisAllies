# Axis & Allies 1940 Global 2E — Game State & Action Space Reference

## Part 1: Architecture Overview

### Two-Level Graph Transformer

**Level 1 — Unit Entity Encoder**
Each territory has up to 20 `FUnitEntity` entries, each encoded as a 25-float vector. A transformer encoder with 2 layers and hidden dimension 64 aggregates these into a single 64-dimensional representation per territory.

**Level 2 — Territory Graph Transformer**
Node features (20 floats) concatenated with entity encoding (64 floats) = 84 floats per territory, projected into 192-dimensional hidden space. Six graph transformer blocks with edge-masked attention produce 13 policy head outputs and a value output.

### State Buffer Layout
```
[0 .. 6579]    Node features: 329 territories x 20 floats = 6,580 floats
[6580 .. 7172] Global features: 593 floats
Total: 7,173 floats
```

### Tensor Shapes (per inference call)
| Input | Shape | Notes |
|---|---|---|
| node_features | [B, 329, 20] | |
| global_features | [B, 593] | B x 593 |
| unit_entities | [B, 329, 20, 25] | max 20 entities per territory |
| entity_counts | [B, 329, 1] | |

---

## Part 2: Territory Layout

- **Total nodes:** 329
- **Land territories:** indices 0-201
- **Sea zones:** indices 202-328
- **Sea zone formula:** SZ_n = territory index (n + 201). SZ1=T202, SZ6=T207, SZ10=T211, etc.

### Players (9, ordered by turn)
| Index | Power |
|---|---|
| 0 | Germany |
| 1 | Soviet Union |
| 2 | Japan |
| 3 | United States |
| 4 | China |
| 5 | United Kingdom |
| 6 | Italy |
| 7 | ANZAC |
| 8 | France |

**Neutral players** (node features [1] and [2] only): 9=StrictNeutral, 10=ProAllies, 11=ProAxis

**Controller encoding:** `(player_index + 1) / 12`. Value 0.0 = uncontrolled.

---

## Part 3: Node Features (20 floats per territory)

**Access:** `state_buffer[territory_index * 20 + feature_index]`

| Index | Feature | Encoding | Notes |
|---|---|---|---|
| [0] | IPC value | raw_IPC / 20 | 0 for sea zones |
| [1] | Current controller | (player + 1) / 12 | Players 0-11 |
| [2] | Original controller | (player + 1) / 12 | Players 0-11 |
| [3] | Minor IC present | binary | |
| [4] | Major IC present | binary | |
| [5] | IC damage | damage / 20 | Max 20 major, 6 minor |
| [6] | Air base present | binary | |
| [7] | Air base damage | damage / 6 | Max 6 |
| [8] | Naval base present | binary | |
| [9] | Naval base damage | damage / 6 | Max 6 |
| [10] | Is sea zone | binary | Static |
| [11] | Is victory city | binary | Static |
| [12] | Is capital territory | binary | Static |
| [13] | Is canal-controlling territory | binary | Static |
| [14] | Is convoy zone | binary | Static |
| [15] | Is kamikaze zone | binary | Static |
| [16] | Scramble capable | binary | Dynamic: air base present AND undamaged |
| [17] | Is island | binary | Static |
| [18] | Is coastal | binary | Static |
| [19] | Start of turn controller | (player + 1) / 12 | Set at turn start; never modified during simulation |

### Quick Node Feature Calculations
```cpp
// Get node feature
float GetNodeFeature(int32 TerritoryIndex, int32 FeatureIndex)
    { return StateBuffer[TerritoryIndex * 20 + FeatureIndex]; }

// IC damage: encode/decode
float EncodeICDamage(int32 Damage)  { return (float)Damage / 20.0f; }
int32 DecodeICDamage(float Encoded) { return FMath::RoundToInt(Encoded * 20.0f); }

// Air/naval base damage: encode/decode
float EncodeBaseDamage(int32 Damage)  { return (float)Damage / 6.0f; }
int32 DecodeBaseDamage(float Encoded) { return FMath::RoundToInt(Encoded * 6.0f); }

// Controller: encode/decode
float EncodePlayer(int32 PlayerIndex) { return (float)(PlayerIndex + 1) / 12.0f; }
int32 DecodePlayer(float Encoded)     { return FMath::RoundToInt(Encoded * 12.0f) - 1; }
```

### Static Territory Classification

**Island territories** ([17]=1, 37 total): West Indies(7), Sardinia(44), Sicily(45), Malta(46), Crete(59), Cyprus(60), French Madagascar(126), Ceylon(131), Sumatra(135), Borneo(136), Java(137), Celebes(138), Dutch New Guinea(139), New Guinea(140), Hainan(159), Formosa(160), Okinawa(161), Iwo Jima(171), Marianas(172), Guam(173), Paulau Island(174), Caroline Islands(175), Marshall Islands(176), Wake Island(177), Midway(178), Johnston Island(179), Hawaiian Islands(180), Gilbert Islands(181), Solomon Islands(182), New Britain(183), New Hebrides(184), Fiji(185), New Zealand(186), Aleutian Islands(196), Samoa(199), Line Islands(200), Philippines(201)

**Inland territories** ([17]=0, [18]=0, 45 total): Bolivia(22), Paraguay(23), France(35), Switzerland(41), Greater Southern Germany(49), Eastern Poland(51), Slovakia Hungary(52), Vologda(67), Belarus(69), Smolensk(70), Russia(71), Bryansk(72), Tambov(73), Samara(74), Volgograd(75), Western Ukraine(76), Urals(80), Novosibirsk(82), Timguska(83), Yenisey(84), Yakut S.S.R.(85), Buryatia(86), Sakha(89), Afghanistan(99), French Central Africa(116), Rhodesia(123), Tsinghai(141), Sikang(142), Kansu(143), Shensi(144), Szechwan(145), Kweichow(146), Yunnan(147), Hunan(148), Hopei(151), Anhwe(152), Suiyuyan(154), Chahar(155), Olgiy(162), Tsagaan-Olom(163), Dzavhan(164), Central Mongolia(165), Ulaanbaatar(166), Buyant-Uhaa(167), Evenkiyskiy(195)

**Coastal territories** ([17]=0, [18]=1): all remaining land territories (120 total)

**Sea zones** ([10]=1, [17]=0, [18]=0): indices 202-328

---

## Part 4: Unit Entity Features (25 floats per entity)

**Access:** `entity_tensor[batch, territory, entity_slot, feature_index]`

| Index | Feature | Encoding | Notes |
|---|---|---|---|
| [0] | Unit type | unit_type / 13 | See unit type table |
| [1] | Owning player | (player + 1) / 12 | |
| [2] | Count | min(count, 20) / 20 | Identical units merged |
| [3] | Hit points | hp / 2 | BB/CV undamaged = 1.0 |
| [4] | Movement remaining | min(movement, 8) / 8 | |
| [5] | Slot ID | slot_id / 20 | Transport/carrier unique ID; 0 for others |
| [6] | Is scrambled | binary | |
| [7] | Has loaded this turn | binary | |
| [8] | Has unloaded this turn | binary | |
| [9] | Is submerged | binary | |
| [10] | Combat engagement state | value / 2 | If !bIsStrategicBombing: 0=default, 1=has engaged in combat; if bIsStrategicBombing: 0=targeting IC, 1=targeting air base, 2=targeting naval base |
| [11] | Is retreating | binary | |
| [12] | Cargo unit type A | (type + 1) / 14; 0 if empty | |
| [13] | Cargo unit owner A | (owner + 1) / 12; 0 if empty | |
| [14] | Cargo unit type B | (type + 1) / 14; 0 if empty | |
| [15] | Cargo unit owner B | (owner + 1) / 12; 0 if empty | |
| [16] | Is strategic bombing | binary | Committed to SBR this turn |
| [17] | Is escorting | binary | Committed as SBR escort |
| [18] | Is intercepting | binary | Committed as SBR interceptor |
| [19] | Is conducting surprise strike | binary | Sub firing surprise strike |
| [20] | IsBombarding (int32) | value / 329.0f | 0=not bombarding/not offloading; >0=destination territory index+1. Battleships/cruisers: bombard destination. Transports: offload destination declaration (mutually exclusive by unit type). |
| [21] | Has completed surprise strike | binary | Persists full turn |
| [22] | Has completed bombardment | binary | Persists full turn |
| [23] | Is paratrooper | binary | Subject to AA fire round 0 |
| [24] | Start of turn territory | territory_index / 328 | Territory at start of turn; used by offloaded land units to identify source sea zone for bombardment eligibility |

**IsBombarding dual-use note:** `IsBombarding` is an int32 (not bool). For battleships/cruisers, >0 encodes the bombardment destination territory+1. For transports, >0 encodes the offload destination territory+1. The two uses are mutually exclusive by unit type. Blueprint uses StartOfTurnTerritory on offloaded land units to identify which sea zone the transport came from, enabling correct bombardment eligibility enforcement.

**Cargo encoding note:** In the C++ struct and JSON, cargo types are stored as raw integers (-1 = empty, 0-13 = unit type). `ToModelFeatures` converts to `(type + 1) / 14.0f` with 0.0f for empty. Same pattern for cargo owner: -1 = empty, encoded as 0.0f.

### Unit Type Indices
| Index | Unit | Attack | Defense | Move | HP | Cost |
|---|---|---|---|---|---|---|
| 0 | Infantry | 1 (2 w/art) | 2 | 1 | 1 | 3 |
| 1 | Artillery | 2 | 2 | 1 | 1 | 4 |
| 2 | Mechanized Infantry | 1 (2 w/art or tank) | 2 | 2 | 1 | 4 |
| 3 | Tank | 3 | 3 | 2 | 1 | 6 |
| 4 | Fighter | 3 (4 w/Jet) | 4 | 4 | 1 | 10 |
| 5 | Tactical Bomber | 3 (4 w/Ftr or Tank) | 3 | 4 | 1 | 11 |
| 6 | Strategic Bomber | 4 | 1 | 6 | 1 | 12 |
| 7 | AA Gun | 0 | 0 | 1 | 1 | 5 |
| 8 | Submarine | 2 (3 w/SuperSubs) | 1 | 2 | 1 | 6 |
| 9 | Destroyer | 2 | 2 | 2 | 1 | 8 |
| 10 | Cruiser | 3 | 3 | 2 | 1 | 12 |
| 11 | Battleship | 4 | 4 | 2 | 2 | 20 |
| 12 | Carrier | 0 | 2 | 2 | 2 | 16 |
| 13 | Transport | 0 | 0 | 2 | 1 | 7 |

**HP at game start:** Battleship=2, Carrier=2, all others=1. Ships repair automatically at naval bases; no player decision required.

**Air units on carriers:** represented both as top-level entity entries (SlotId = carrier's SlotId) AND in carrier's CargoUnitTypeA/B fields. Both maintained simultaneously.

**Loaded ground units on transports:** cargo fields only — no top-level entity entry.

**Combined arms (attacker only):**
- Infantry + Artillery: Infantry attacks at 2 (each artillery supports one infantry; Advanced Artillery: one artillery supports two)
- Mechanized Infantry + Artillery: Mech Infantry attacks at 2
- Tactical Bomber + Tank: Tactical Bomber attacks at 4
- Tactical Bomber + Fighter: Tactical Bomber attacks at 4
- Each unit may be paired with only one other unit at a time
- With Improved Mech Infantry tech: Mech Infantry attacks at 2 when paired with tank or artillery, and can blitz without a tank

---

## Part 5: Global Features (593 floats)

**Access:** `state_buffer[6580 + global_feature_index]`

### Layout Overview
| Range | Content | Size |
|---|---|---|
| [0-2] | Turn context | 3 |
| [3-38] | Economy (4 values x 9 players) | 36 |
| [39-146] | Technology (12 techs x 9 players) | 108 |
| [147-155] | Research investment (9 players) | 9 |
| [156-192] | National objectives (37 total) | 37 |
| [193-205] | Victory state | 13 |
| [206-214] | Capital status (9 players) | 9 |
| [215-250] | War status (36 pairs) | 36 |
| [251-253] | Neutral territory state | 3 |
| [254-379] | Global unit totals (14 types x 9 players) | 126 |
| [380-550] | Mobilization queue (19 types x 9 players) | 171 |
| [551-556] | Special state flags | 6 |
| [557] | kamikaze_remaining | 1 |
| [558-592] | Active battle state | 35 |

### Turn Context [0-2]
| Index | Feature | Encoding |
|---|---|---|
| [0] | PhaseId | phase_id / 50 |
| [1] | Current player | player / 8 |
| [2] | Current turn | turn / 40 |

### Economy [3-38]
Base = `3 + player_index * 4`

| Offset | Feature | Encoding |
|---|---|---|
| +0 | Treasury | IPC / 200 |
| +1 | Income | IPC / 100 |
| +2 | Convoy disruption loss | IPC / 100 |
| +3 | Production capacity | count / 100 |

```cpp
int32 TreasuryIndex(int32 Player) { return 6580 + 3 + Player * 4; }
int32 IncomeIndex(int32 Player)   { return 6580 + 4 + Player * 4; }
```

### Technology [39-146]
`GLOBAL_TECH_OFFSET = 39`, `TECHS_PER_PLAYER = 12`

Access: `global_features[39 + player_index * 12 + tech_index]`
Flat: `state_buffer[6619 + player_index * 12 + tech_index]`

| Tech Index | Technology |
|---|---|
| 0 | Long Range Aircraft |
| 1 | Improved Shipyards |
| 2 | Radar |
| 3 | Rockets |
| 4 | Improved Mechanized Infantry |
| 5 | Advanced Artillery |
| 6 | Super Submarines |
| 7 | Jet Fighters |
| 8 | War Bonds |
| 9 | Combined Bombardment |
| 10 | Paratroopers |
| 11 | Heavy Bombers |

```cpp
int32 TechFlatIndex(int32 Player, int32 Tech) { return 6619 + Player * 12 + Tech; }
bool HasTech(const TArray<float>& StateBuffer, int32 Player, int32 Tech)
    { return StateBuffer[6619 + Player * 12 + Tech] > 0.5f; }
```

### National Objectives [156-192] (37 total)
One float per national objective at indices 156-192. Blueprint is entirely responsible for defining which slot maps to which objective and for writing the correct value each turn. C++ and the model treat these as 37 anonymous features.

Most slots are binary (0 or 1). Exception: slot index 11 (global index 167) is an integer divided by 40, representing a count of Pro-Axis or Pro-Allies territories controlled by the USSR.

Initial values at game start:
- Index 0 (slot 156): 1.0
- Index 10 (slot 166): 1.0
- Index 11 (slot 167): 0.0 (count / 40, starting count = 0)
- Index 13 (slot 169): 1.0
- Index 25 (slot 181): 1.0
- All others: 0.0

### Victory State [193-205]
| Index | Feature | Encoding |
|---|---|---|
| [193-201] | Victory city count per player (players 0-8) | count / 50 |
| [202] | Total Axis victory cities | count / 50 |
| [203] | Total Allied victory cities | count / 50 |
| [204] | Allied won | binary |
| [205] | Axis won | binary |

### Capital Status [206-214]
| Index | Feature | Encoding |
|---|---|---|
| [206-214] | Capital held per player (players 0-8) | binary (1.0 = held, 0.0 = captured) |

### War Status [215-250]
36 pairs for 9 players. Index for pair (I < J): `215 + (I * (17 - I) / 2) + (J - I - 1)`

```cpp
int32 WarStatusFlatIndex(int32 PlayerA, int32 PlayerB)
{
    int32 I = FMath::Min(PlayerA, PlayerB);
    int32 J = FMath::Max(PlayerA, PlayerB);
    return 6580 + 215 + ((I * (17 - I)) / 2) + (J - I - 1);
}
```

### Special State Flags [551-556]
| Index | Feature | Notes |
|---|---|---|
| [551] | Pending sub-action value | Written by C++ on chain-final nodes |
| [552] | USSR attacked Mongolia-adjacent | binary |
| [553] | Japan trade bonus permanently lost | binary |
| [554] | France liberation bonus used | binary |
| [555] | USA forced entry available | binary |
| [556] | USA has entered the war | binary |

### GlobalFeatures[551] — Pending Sub-Action Conditioning
Written by C++ on chain-final child nodes. Value = `action / preceding_head_size`.

| Chain-final phase | Preceding head size |
|---|---|
| TechChartSelection (1) | 7 |
| RepairQuantity (3) | 988 |
| PurchaseQuantity (5) | 20 |
| KamikazeTarget (8) | 128 |
| ScrambleQuantity (11) | 128 |
| BombardQuantity (14) | 202 |
| CombatMoveQuantity (19) | 330 |
| StrategicBombingQuantity (22) | 3 |
| SubmarineActionQuantity (25) | 2 |
| CombatResolveCasualtyQuantity (27) | 14 |
| NonCombatQuantity (32) | 330 |
| PlacementQuantity (34) | 330 or 6581 |
| SBR_InterceptorQuantity (36) | 6581 |
| SBR_AirBattleCasualtyQuantity (38) | 14 |
| SBR_EscortQuantity (40) | 6581 |
| AirUnitLandOnCarrierCount (45) | 20 |

### kamikaze_remaining [557]
Encoding: `count / 6`. Japan starts at 1.0 (6 strikes). Flat index: `6580 + 557 = 7137`.

### Active Battle State [558-592]
All zeros when no battle active (ROUND_NUMBER = 0 signals no active battle). Blueprint writes when battle begins; clears when battle ends.

**Units excluded when building battle state:** any entity with `bIsStrategicBombing`, `bIsEscorting`, `bIsIntercepting`, `bIsConductingSurpriseStrike`, `IsBombarding > 0`, `bHasCompletedSurpriseStrike`, or `bHasCompletedBombardment` = true.

| Index | Feature | Encoding |
|---|---|---|
| [558] | Territory ID | territory_index / 328 |
| [559] | Attacker | (player + 1) / 12 |
| [560] | Defender | (player + 1) / 12 |
| [561-574] | Attacking unit counts (14 types) | count / 20 |
| [575-588] | Defending unit counts (14 types) | count / 20 |
| [589] | ATK_HITS_PENDING | hits / 280 |
| [590] | DEF_HITS_PENDING | hits / 280 |
| [591] | ROUND_NUMBER | round / 20 |
| [592] | SEABORNE | binary |

Unit count order within [561-574] and [575-588] follows unit type indices 0-13.

**ROUND_NUMBER note:** 0 = no active battle. Blueprint sets to 0 when clearing battle state. Minimum valid round during combat = 1 (encoded as 1/20 = 0.05).

**ATK_HITS / DEF_HITS note:** Maximum possible hits = 14 unit types x 20 count = 280. Encoded as hits / 280.0f.

```cpp
int32 BattleTerritoryFlatIndex()              { return 6580 + 558; }  // = 7138
int32 BattleAtkUnitFlatIndex(int32 UnitType)  { return 6580 + 561 + UnitType; }
int32 BattleDefUnitFlatIndex(int32 UnitType)  { return 6580 + 575 + UnitType; }
int32 BattleAtkHitsFlatIndex()                { return 6580 + 589; }  // = 7169
int32 BattleDefHitsFlatIndex()                { return 6580 + 590; }  // = 7170

float EncodeHits(int32 Hits)    { return FMath::Clamp((float)Hits / 280.0f, 0.0f, 1.0f); }
int32 DecodeHits(float Encoded) { return FMath::RoundToInt(Encoded * 280.0f); }
float EncodeRound(int32 Round)  { return (float)Round / 20.0f; }
int32 DecodeRound(float Encoded){ return FMath::RoundToInt(Encoded * 20.0f); }
```

### Quick Global Feature Access
```cpp
float GetGlobal(const TArray<float>& Buf, int32 GlobalIndex)
    { return Buf[6580 + GlobalIndex]; }
void SetGlobal(TArray<float>& Buf, int32 GlobalIndex, float Value)
    { Buf[6580 + GlobalIndex] = Value; }
```

---

## Part 6: Phase Table (46 phases, Count=46)

| PhaseId | Name | Head | Size | Encoding | Pass | Chain Role |
|---|---|---|---|---|---|---|
| 0 | Tech | Head7 | 7 | 0=Pass, 1-6 dice | 0 | Standalone |
| 1 | TechChartSelection | Head2 | 2 | 0=Chart A, 1=Chart B | - | Chain-final |
| 2 | UnitRepair | Head988 | 988 | territory * 3 + repair_type; 0=Pass | 0 | SimulateTransition |
| 3 | RepairQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 4 | PurchaseType | Head20 | 20 | 0=Pass, 1-14=unit, 15-19=facility | 0 | SimulateTransition |
| 5 | PurchaseQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 6 | KamikazeQuantity | Head7 | 7 | 0=Pass, 1-6 strikes | 0 | SimulateTransition |
| 7 | KamikazeLocation | Head128 | 128 | sea zone 0-126; 127=N/A | - | Chain-intermediate |
| 8 | KamikazeTarget | Head20 | 20 | slot_id 0-19 | - | Chain-final |
| 9 | ScrambleSource | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 10 | ScrambleUnitDest | Head128 | 128 | sea zone 0-126; 127=N/A | - | Chain-intermediate |
| 11 | ScrambleQuantity | Head3 | 3 | quantity 1-3 | - | Chain-final |
| 12 | BombardSource | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 13 | BombardDest | Head202 | 202 | territory 0-201 | - | Chain-intermediate |
| 14 | BombardQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 15 | DeclareWar | Head10 | 10 | 0=Pass, 1-9=player | 0 | Standalone |
| 16 | CombatMoveSource | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 17 | LoadUnitsCombat | Head49 | 49 | see LoadUnits encoding; 0=Pass | 0 | SimulateTransition |
| 18 | CombatMoveUnitDest | Head330 | 330 | territory 0-328; 0=Pass | 0 | Chain-intermediate |
| 19 | CombatMoveQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 20 | StrategicBombingSource | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 21 | StrategicBombingDecision | Head3 | 3 | 0=IC, 1=air base, 2=naval base | - | Chain-intermediate |
| 22 | StrategicBombingQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 23 | SubmarineActionSource | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 24 | SubmarineActionType | Head2 | 2 | 0=submerge, 1=surprise strike | - | Chain-intermediate |
| 25 | SubmarineActionQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 26 | CombatResolveCasualtyType | Head14 | 14 | unit type 0-13 | - | Chain-intermediate |
| 27 | CombatResolveCasualtyQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 28 | CombatResolveContinueOrRetreat | Head13 | 13 | 0=continue, 1-12=retreat slot | 0 | Standalone |
| 29 | NonCombatSource | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 30 | LoadUnitsNonCombat | Head49 | 49 | see LoadUnits encoding; 0=Pass | 0 | SimulateTransition |
| 31 | NonCombatUnitDest | Head330 | 330 | territory 0-328; 0=Pass | 0 | Chain-intermediate |
| 32 | NonCombatQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 33 | PlacementTerritory | Head330 | 330 | territory 0-328 | - | Chain-intermediate |
| 34 | PlacementQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 35 | SBR_InterceptorCommitment | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 36 | SBR_InterceptorQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 37 | SBR_AirBattleCasualtyType | Head14 | 14 | unit type 0-13 | - | Chain-intermediate |
| 38 | SBR_AirBattleCasualtyQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 39 | SBR_EscortCommitment | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 40 | SBR_EscortQuantity | Head20 | 20 | quantity - 1 | - | Chain-final |
| 41 | PlacementCarrier | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | Chain-intermediate |
| 42 | AirUnitLandOnCarrier | Head6581 | 6581 | territory * 20 + slot_id; 0=Pass | 0 | SimulateTransition |
| 43 | AirUnitLandOnCarrierDest | Head20 | 20 | carrier slot_id 0-19 | - | Chain-intermediate |
| 44 | AirUnitLandOnCarrierPlaneQuantity | Head20 | 20 | quantity - 1 | - | Chain-intermediate |
| 45 | AirUnitLandOnCarrierCount | Head20 | 20 | quantity - 1 | - | Chain-final |

### Chain Role Key
- **Standalone:** No pending context. Blueprint routes to next phase entirely.
- **SimulateTransition:** Has pass option. C++ calls Blueprint SimulateTransition. Sets PendingActionA if non-pass.
- **Chain-intermediate:** No SimulateTransition. C++ copies state forward, routes to next phase via GetNextChainPhase.
- **Chain-final:** C++ calls Blueprint SimulateTransition with full pending context (A, B, C, D).

### Chain Sequences
| Chain | Steps |
|---|---|
| Tech | Tech(SimTrans) -> TechChartSelection(final) |
| UnitRepair | UnitRepair(SimTrans) -> RepairQuantity(final) |
| Purchase | PurchaseType(SimTrans) -> PurchaseQuantity(final) |
| Kamikaze | KamikazeQuantity(SimTrans) -> KamikazeLocation(intermediate) -> KamikazeTarget(final) |
| Scramble | ScrambleSource(SimTrans) -> ScrambleUnitDest(intermediate) -> ScrambleQuantity(final) |
| Bombard | BombardSource(SimTrans) -> BombardDest(intermediate) -> BombardQuantity(final) |
| CombatMove | CombatMoveSource(SimTrans) -> [LoadUnitsCombat(SimTrans) x0-2] -> CombatMoveUnitDest(intermediate) -> CombatMoveQuantity(final) |
| StrategicBombing | StrategicBombingSource(SimTrans) -> StrategicBombingDecision(intermediate) -> StrategicBombingQuantity(final) |
| SubmarineAction | SubmarineActionSource(SimTrans) -> SubmarineActionType(intermediate) -> SubmarineActionQuantity(final) |
| CombatCasualty | CombatResolveCasualtyType(intermediate) -> CombatResolveCasualtyQuantity(final) |
| NonCombatMove | NonCombatSource(SimTrans) -> [LoadUnitsNonCombat(SimTrans) x0-2] -> NonCombatUnitDest(intermediate) -> NonCombatQuantity(final) |
| Placement | PlacementTerritory(intermediate) -> PlacementQuantity(final) |
| PlacementCarrier | PlacementCarrier(intermediate) -> PlacementQuantity(final) |
| SBR_Interceptor | SBR_InterceptorCommitment(SimTrans) -> SBR_InterceptorQuantity(final) |
| SBR_AirBattle | SBR_AirBattleCasualtyType(intermediate) -> SBR_AirBattleCasualtyQuantity(final) |
| SBR_Escort | SBR_EscortCommitment(SimTrans) -> SBR_EscortQuantity(final) |
| AirLandOnCarrier | AirUnitLandOnCarrier(SimTrans) -> AirUnitLandOnCarrierDest(intermediate) -> AirUnitLandOnCarrierPlaneQuantity(intermediate) -> AirUnitLandOnCarrierCount(final) |

### Policy Heads (13 active heads)
| Head | Size | Phases Using It |
|---|---|---|
| Head2 | 2 | TechChartSelection(1), SubmarineActionType(24) |
| Head3 | 3 | ScrambleQuantity(11), StrategicBombingDecision(21) |
| Head7 | 7 | Tech(0), KamikazeQuantity(6) |
| Head10 | 10 | DeclareWar(15) |
| Head13 | 13 | CombatResolveContinueOrRetreat(28) |
| Head14 | 14 | CombatResolveCasualtyType(26), SBR_AirBattleCasualtyType(37) |
| Head20 | 20 | RepairQuantity(3), PurchaseType(4), PurchaseQuantity(5), KamikazeTarget(8), BombardQuantity(14), CombatMoveQuantity(19), StrategicBombingQuantity(22), SubmarineActionQuantity(25), CombatResolveCasualtyQuantity(27), NonCombatQuantity(32), PlacementQuantity(34), SBR_InterceptorQuantity(36), SBR_AirBattleCasualtyQuantity(38), SBR_EscortQuantity(40), AirUnitLandOnCarrierDest(43), AirUnitLandOnCarrierPlaneQuantity(44), AirUnitLandOnCarrierCount(45) |
| Head49 | 49 | LoadUnitsCombat(17), LoadUnitsNonCombat(30) |
| Head128 | 128 | KamikazeLocation(7), ScrambleUnitDest(10) |
| Head202 | 202 | BombardDest(13) |
| Head330 | 330 | CombatMoveUnitDest(18), NonCombatUnitDest(31), PlacementTerritory(33) |
| Head6581 | 6581 | ScrambleSource(9), BombardSource(12), CombatMoveSource(16), StrategicBombingSource(20), SubmarineActionSource(23), SBR_InterceptorCommitment(35), SBR_EscortCommitment(39), PlacementCarrier(41), AirUnitLandOnCarrier(42) |
| Head988 | 988 | UnitRepair(2) |

---

## Part 7: Action Space Encodings (per phase)

### Head6581 phases — territory * 20 + slot_id encoding
Used by: ScrambleSource(9), BombardSource(12), CombatMoveSource(16), StrategicBombingSource(20), SubmarineActionSource(23), NonCombatSource(29), SBR_InterceptorCommitment(35), SBR_EscortCommitment(39), PlacementCarrier(41), AirUnitLandOnCarrier(42).

Action 0 = Pass (except PlacementCarrier and AirUnitLandOnCarrier which have no pass).

```cpp
int32 EncodeHead6581(int32 Territory, int32 SlotId) { return Territory * 20 + SlotId + 1; }  // +1 for pass offset
int32 DecodeTerritory(int32 Action) { return (Action - 1) / 20; }
int32 DecodeSlotId(int32 Action)    { return (Action - 1) % 20; }
```

### Tech (0) — Head7, size 7
| Action | Meaning |
|---|---|
| 0 | Pass (no tech purchase) |
| 1-6 | Purchase N tech dice |

### TechChartSelection (1) — Head2, size 2
| Action | Meaning |
|---|---|
| 0 | Advance tech on Chart A |
| 1 | Advance tech on Chart B |

### UnitRepair (2) — Head988, size 988
Encoding: `territory * 3 + repair_type + 1`. Pass = 0.

| repair_type | Meaning |
|---|---|
| 0 | Repair IC at territory |
| 1 | Repair air base at territory |
| 2 | Repair naval base at territory |

```cpp
int32 EncodeUnitRepair(int32 Territory, int32 RepairType) { return Territory * 3 + RepairType + 1; }
int32 DecodeUnitRepairTerritory(int32 Action) { return (Action - 1) / 3; }
int32 DecodeUnitRepairType(int32 Action)      { return (Action - 1) % 3; }
```

### RepairQuantity (3) — Head20, size 20
Quantity of IPC spent (1-20). Action = quantity - 1.

### PurchaseType (4) — Head20, size 20
| Action | Meaning |
|---|---|
| 0 | Pass (done purchasing) |
| 1-14 | Unit type index 0-13 (see Part 4) |
| 15 | Minor IC |
| 16 | Major IC |
| 17 | Air base |
| 18 | Naval base |
| 19 | IC Upgrade (minor -> major) |

### PurchaseQuantity (5) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### KamikazeQuantity (6) — Head7, size 7
| Action | Meaning |
|---|---|
| 0 | Pass (no more strikes) |
| 1-6 | Number of strikes to commit |

### KamikazeLocation (7) — Head128, size 128
Sea zone 0-126. Action 127 = N/A. No pass option.
Kamikaze-eligible sea zones: SZ6(T207), SZ17(T218), SZ19(T220), SZ20(T221), SZ22(T223), SZ35(T236).

### KamikazeTarget (8) — Head20, size 20
Target ship slot_id 0-19 in selected sea zone. No pass option.

### ScrambleSource (9) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0. Selects scrambling unit and source territory.

### ScrambleUnitDest (10) — Head128, size 128
Destination sea zone 0-126. Action 127 = N/A. No pass option.

### ScrambleQuantity (11) — Head3, size 3
| Action | Meaning |
|---|---|
| 0 | Scramble 1 unit |
| 1 | Scramble 2 units |
| 2 | Scramble 3 units |

### BombardSource (12) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0. Selects bombarding ship and source sea zone.

### BombardDest (13) — Head202, size 202
Destination land territory 0-201. No pass option.

### BombardQuantity (14) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### DeclareWar (15) — Head10, size 10
| Action | Meaning |
|---|---|
| 0 | Pass |
| 1-9 | Declare war on player (player = action - 1) |

### CombatMoveSource (16) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0.

### LoadUnitsCombat (17) / LoadUnitsNonCombat (30) — Head49, size 49
| Action | Meaning |
|---|---|
| 0 | Pass |
| 1-9 | Load Infantry from player 0-8 |
| 10-18 | Load Artillery from player 0-8 |
| 19-27 | Load Mech Infantry from player 0-8 |
| 28-36 | Load Tank from player 0-8 |
| 37-45 | Load AA Gun from player 0-8 |
| 46 | Unload cargo slot A |
| 47 | Unload cargo slot B |
| 48 | Unload both cargo slots |

```cpp
// Decode load action (actions 1-45 only)
int32 UnitTypeOffset = (Action - 1) / 9;  // 0=Inf, 1=Art, 2=Mech, 3=Tank, 4=AA
int32 Player         = (Action - 1) % 9;  // 0-8
```

### CombatMoveUnitDest (18) / NonCombatUnitDest (31) — Head330, size 330
| Action | Meaning |
|---|---|
| 0 | Pass |
| 1-329 | Destination territory = action - 1 |

### CombatMoveQuantity (19) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### StrategicBombingSource (20) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0. Selects the bombing unit and its source territory.

### StrategicBombingDecision (21) — Head3, size 3
| Action | Meaning |
|---|---|
| 0 | Target IC (strategic bombers only) |
| 1 | Target air base |
| 2 | Target naval base |

No pass option.

### StrategicBombingQuantity (22) — Head20, size 20
Number of bombers committed (1-20). Action = quantity - 1.

### SubmarineActionSource (23) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0. Pass means submarine engages in regular combat (no further decision needed).

### SubmarineActionType (24) — Head2, size 2
| Action | Meaning |
|---|---|
| 0 | Submerge |
| 1 | Surprise strike |

No pass option. Only legal when no enemy destroyer is present.

### SubmarineActionQuantity (25) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### CombatResolveCasualtyType (26) — Head14, size 14
Unit type index 0-13. No pass option. Blueprint enforces only types present in battle state are legal.

### CombatResolveCasualtyQuantity (27) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### CombatResolveContinueOrRetreat (28) — Head13, size 13
| Action | Meaning |
|---|---|
| 0 | Continue combat |
| 1-12 | Retreat to destination slot |

Retreat slot interpretation depends on GLOBAL_BATTLE_SEABORNE:
- SEABORNE=0 (land battle): retreat slot refers to adjacent land territory
- SEABORNE=1 (sea battle): retreat slot refers to adjacent sea zone

Seaborne units in amphibious assault cannot retreat.

### NonCombatSource (29) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0.

### NonCombatQuantity (32) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### PlacementTerritory (33) — Head330, size 330
Destination territory 0-328 for land unit and facility placement. No pass option.

### PlacementQuantity (34) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### SBR_InterceptorCommitment (35) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0. Defender selects fighter to commit as interceptor.

### SBR_InterceptorQuantity (36) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### SBR_AirBattleCasualtyType (37) — Head14, size 14
Unit type index 0-13. No pass option.

### SBR_AirBattleCasualtyQuantity (38) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### SBR_EscortCommitment (39) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0. Attacker selects fighter to commit as escort.

### SBR_EscortQuantity (40) — Head20, size 20
Quantity (1-20). Action = quantity - 1.

### PlacementCarrier (41) — Head6581, size 6581
territory * 20 + slot_id. No pass option. Selects carrier for fighter/tactical bomber placement during Mobilize phase.

### AirUnitLandOnCarrier (42) — Head6581, size 6581
territory * 20 + slot_id. Pass = 0. Selects which plane entity is landing on a carrier.

### AirUnitLandOnCarrierDest (43) — Head20, size 20
Carrier slot_id 0-19 (same sea zone as selected plane). No pass option.

### AirUnitLandOnCarrierPlaneQuantity (44) — Head20, size 20
How many planes from the entity to land (1-20). Action = quantity - 1.

### AirUnitLandOnCarrierCount (45) — Head20, size 20
How many carriers from the carrier entity to distribute planes across (1-20). Action = quantity - 1.

**AirUnitLandOnCarrier distribution logic:**
```cpp
int32 PlaneCount   = PlaneQuantityAction + 1;
int32 CarrierCount = CarrierCountAction + 1;
int32 PlanesPerCarrier = PlaneCount / CarrierCount;     // integer division; always 1 or 2
int32 Remainder        = PlaneCount % CarrierCount;

// Remainder carriers get PlanesPerCarrier + 1 planes
// CarrierCount - Remainder carriers get PlanesPerCarrier planes
// If Remainder == 0: all carriers get PlanesPerCarrier planes
// If PlanesPerCarrier == 1 && Remainder > 0: Remainder carriers get 2 planes
// If PlanesPerCarrier == 2 && Remainder == 0: all carriers get 2 planes
```

---

## Part 8: Architecture Constants

```
NUM_TERRITORIES              = 329
NODE_FEATURE_COUNT           = 20
UNIT_ENTITY_FEATURE_COUNT    = 25
MAX_UNIT_ENTITIES_PER_NODE   = 20
ENTITY_HIDDEN_DIM            = 64
GLOBAL_FEATURE_COUNT         = 593
NUM_PLAYERS                  = 9
NUM_PHASES                   = 46  (EPhaseId::Count = 46)
NUM_POLICY_HEADS             = 13
NUM_UNIT_TYPES               = 14
NUM_MOB_TYPES                = 19
NUM_TECHNOLOGIES             = 12
MAX_SLOT_ID                  = 19
MAX_CONTAINER_UNITS          = 20
MAX_INFERENCE_BATCH_SIZE     = 512
GLOBAL_TECH_OFFSET           = 39
TECHS_PER_PLAYER             = 12
GLOBAL_KAMIKAZE_REMAINING    = 557   // global index; flat = 7137
GLOBAL_BATTLE_TERRITORY_ID   = 558   // flat = 7138
GLOBAL_BATTLE_ATTACKER       = 559
GLOBAL_BATTLE_DEFENDER       = 560
GLOBAL_BATTLE_ATK_UNITS      = 561   // 14 floats [561-574]
GLOBAL_BATTLE_DEF_UNITS      = 575   // 14 floats [575-588]
GLOBAL_BATTLE_ATK_HITS       = 589   // flat = 7169; encoded as hits/280
GLOBAL_BATTLE_DEF_HITS       = 590   // flat = 7170; encoded as hits/280
GLOBAL_BATTLE_ROUND_NUMBER   = 591   // encoded as round/20; 0 = no active battle
GLOBAL_BATTLE_SEABORNE       = 592   // flat = 7172
NODE_FLAT_SIZE               = 329 * 20 = 6,580
GLOBAL_OFFSET_IN_BUFFER      = 6,580
TOTAL_STATE_BUFFER_SIZE      = 7,173
SEA_ZONE_TERRITORY_OFFSET    = 201   // SZ_n = n + 201
```

---

## Part 9: Key Formula Summary

```cpp
// State buffer access
NodeFeature(T, F)           = state_buffer[T * 20 + F]
GlobalFeature(G)            = state_buffer[6580 + G]

// Player encoding
EncodePlayer(P)             = (P + 1) / 12.0f
DecodePlayer(V)             = round(V * 12.0f) - 1

// Current player encoding (turn context [1] only)
EncodeCurrentPlayer(P)      = P / 8.0f
DecodeCurrentPlayer(V)      = round(V * 8.0f)

// Sea zone territory index
TerritoryOfSZ(SZNumber)     = SZNumber + 201

// Technology
TechFlatIndex(P, T)         = 6619 + P * 12 + T

// Treasury
TreasuryFlatIndex(P)        = 6583 + P * 4

// Mobilization queue
MobQueueFlatIndex(MobType, Player) = 6580 + 380 + MobType * 9 + Player

// War status
WarStatusFlatIndex(I, J)    = 6795 + ((I * (17 - I)) / 2) + (J - I - 1)  // I < J

// Kamikaze remaining
KamikazeFlatIndex()         = 7137
EncodeKamikaze(Count)       = Count / 6.0f

// Battle state
BattleAtkUnitFlat(UnitType) = 6580 + 561 + UnitType
BattleDefUnitFlat(UnitType) = 6580 + 575 + UnitType
EncodeHits(H)               = H / 280.0f
DecodeHits(V)               = round(V * 280.0f)
EncodeRound(R)              = R / 20.0f
DecodeRound(V)              = round(V * 20.0f)
EncodeTerritoryID(T)        = T / 328.0f

// Head6581 action encoding (territory * 20 + slot_id + 1; 0 = pass)
EncodeHead6581(Terr, Slot)  = Terr * 20 + Slot + 1
DecodeTerritory(A)          = (A - 1) / 20
DecodeSlotId(A)             = (A - 1) % 20

// Head330 action encoding (CombatMoveUnitDest, NonCombatUnitDest)
// 0 = pass, 1-329: territory = action - 1

// Head988 UnitRepair
EncodeUnitRepair(Terr, Type) = Terr * 3 + Type + 1
DecodeRepairTerritory(A)     = (A - 1) / 3
DecodeRepairType(A)          = (A - 1) % 3

// IsBombarding (int32 entity feature [20])
// 0 = not bombarding / not offloading
// > 0 = destination territory index + 1
// Encoded in model: IsBombarding / 329.0f
EncodeIsBombarding(TerritoryIdx) = TerritoryIdx + 1
DecodeIsBombarding(Value)        = Value - 1   // territory index

// LoadUnits decode (actions 1-45)
UnitTypeOffset(A) = (A - 1) / 9    // 0=Inf, 1=Art, 2=Mech, 3=Tank, 4=AA
PlayerIndex(A)    = (A - 1) % 9    // 0-8
```

---

## Part 10: Combat Resolution Rules Summary

### SBR Resolution Sequence (per territory)
1. Check for interceptors (bIsIntercepting = true)
2. If interceptors present: air battle (one round, all units attack/defend at 1; escorts + bombers vs interceptors)
3. Facility AA fires at surviving bombers only (not escorts): 1 die per bomber per facility; hits on 1 (2 with Radar)
4. Attacker freely chooses which bombers are removed (no casualty selection phase)
5. Surviving bombers roll bombing dice: 1 die each; +2 for strategic bombers; Heavy Bombers: roll 2, take best
6. Total damage per facility, cap at max (major IC: 20, minor IC: 6, air base: 6, naval base: 6)
7. Add to existing facility damage

### General Combat Sequence (per space)
1. AA gun fire (round 1 only): fires at attacking air units + paratroopers; hits on 1 (2 with Radar); not on the battle strip
2. Submarine surprise strike or submerge (if no enemy destroyer): attacker decides first; attacking subs hit on 2 or less; defending subs hit on 1
3. Attacker regular fire (combined arms bonuses apply)
4. Defender regular fire (base values only, no combined arms)
5. Remove defender casualties
6. Press attack or retreat

### Amphibious Assault Sequence (per sea zone)
1. Sea combat (if defending surface warships or scrambled air units present)
2. Battleship/cruiser bombardment (only if NO sea combat occurred): battleships hit on 4 or less, cruisers on 3 or less
3. Land unit offloading
4. Land combat

### Bombardment Eligibility
- Ship must be in same sea zone as offloading transport
- Number of bombarding ships per territory <= number of land units offloaded into that territory
- Transport's StartOfTurnTerritory identifies eligible bombarding sea zone
- IsBombarding on ship entity encodes destination territory + 1
- IsBombarding on transport entity encodes offload destination territory + 1

### Submarine Rules
- Can pass through sea zones containing enemy warships if no enemy destroyer present
- Must stop when entering sea zone containing enemy destroyer (combat move phase)
- Can move through hostile sea zones in noncombat move, but must stop at destroyer
- Surprise strike and submerge only legal when no enemy destroyer present
- Regular combat always legal
- Submarines cannot hit air units unless friendly destroyer present

### Kamikaze Rules
- Japan only; maximum 6 strikes total game
- Only in sea zones with kamikaze symbol (around Japan, Okinawa, Iwo Jima, Formosa, Marianas, Philippines)
- Only targets surface warships (not submarines or transports)
- All strikes (quantity, location, target) declared before any dice rolled
- Each strike: 1 Kamikaze token spent; hits on 2 or less; hit applied to declared target
- Successful hit prevents bombardment from that sea zone (even if ship survives)
- Multiple strikes may target same ship

### Scramble Rules
- Only fighters and tactical bombers can scramble (not strategic bombers)
- From island or coastal territories with operative air base
- Up to 3 units per territory per sea zone
- Multiple territories adjacent to same sea zone: each may scramble up to 3
- Territory adjacent to multiple sea zones: up to 3 total, split any way
- Allied units may scramble if owning power is at war with attacker, within 3-unit limit
- Scrambled units defend at normal defense values; may not retreat
- After combat: return to source territory; if captured, 1 space to friendly territory or carrier
