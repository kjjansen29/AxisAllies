#include "UAI_ReplayBufferManager.h"

#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/Compression.h"

// ================================================================
// STORAGE LAYOUT (Stage B)
//
// Replay buffer (finished games, final outcome values applied):
//   Saved/AITraining/Replay/index.json            <- GetTotalReplayBufferExportPath()
//   Saved/AITraining/Replay/shard_000001.bin      one file per finished game
//   ...
//   index.json = {"next_shard_id": N,
//                 "shards": [{"file": "shard_000001.bin", "samples": 123}, ...]}
//   Shards are written once and never rewritten. Pruning deletes the
//   oldest shard files. Older .jsonl shards (Stage A) are still read.
//
// Binary shard (.bin), all values little-endian:
//   char[4] "AAR1" | uint32 version (1) | uint32 raw_size | uint32 zlib_size
//   followed by zlib_size bytes of zlib-compressed payload:
//     int32   N
//     int32   phase_id[N]
//     int32   player_id[N]
//     int32   policy_size[N]
//     float32 visit_count[N]
//     float32 value_target[N * NUM_PLAYERS]
//     float32 node_features[N * NUM_TERRITORIES * NODE_FEATURE_COUNT]
//     float32 global_features[N * GLOBAL_FEATURE_COUNT]
//     int32   policy_offsets[N + 1]   int32 policy_index[P]   float32 policy_value[P]
//     int32   legal_offsets[N + 1]    int32 legal_index[L]
//     int32   entity_offsets[N + 1]   int32 entity_territory[M]
//     float32 entity_features[M * UNIT_ENTITY_FEATURE_COUNT]
//   Sample i's sparse entries are [offsets[i], offsets[i + 1]).
//   Entities of a sample are ordered by territory, then slot.
//
// Staging (unfinished games, placeholder value targets):
//   Saved/AITraining/Staging/<Session>.jsonl      append-only, human-readable
//
// Every .jsonl line (staging, and Stage A shards) is one sample:
//   {"phase_id":P,"player_id":Q,"visit_count":V,
//    "policy_size":S,"policy":[[index,prob],...],     <- nonzero entries only
//    "legal_actions":[...],"value_target":[...],
//    "node_features":[...],"global_features":[...],
//    "entities":[[territory,[[25 features],...]],...]} <- territories with units only
//
// Legacy files from before Stage A are still read:
//   Saved/AITraining/TotalReplayBuffer.json (listed in the index as the
//   oldest shard the first time the index is created) and
//   Saved/AITraining/Staging/<Session>.json (merged when the game ends).
// ================================================================

namespace ReplayStorage
{
    struct FShard
    {
        FString File;      // relative to the replay directory
        int32   Samples = 0;
    };

    static FString GetReplayDir()
    {
        return FPaths::GetPath(GetTotalReplayBufferExportPath());
    }

    static FString GetLegacyTotalPath()
    {
        return FPaths::Combine(
            FPaths::ProjectSavedDir(),
            TEXT("AITraining/TotalReplayBuffer.json"));
    }

    static FString GetShardPath(const FString& File)
    {
        return FPaths::Combine(GetReplayDir(), File);
    }

    // ---- Number formatting: compact, always valid JSON ----
    static void AppendNumber(FString& Out, float V)
    {
        if (!FMath::IsFinite(V) || V == 0.0f)
        {
            Out += TEXT("0");
        }
        else if (V == FMath::RoundToFloat(V) && FMath::Abs(V) < 1.0e7f)
        {
            Out.AppendInt((int32)V);
        }
        else
        {
            Out += FString::Printf(TEXT("%.6g"), V);
        }
    }

    static void AppendFloatArray(FString& Out, const TArray<float>& Arr)
    {
        Out += TEXT("[");
        for (int32 i = 0; i < Arr.Num(); i++)
        {
            if (i > 0) Out += TEXT(",");
            AppendNumber(Out, Arr[i]);
        }
        Out += TEXT("]");
    }

    // ---- One sample -> one JSON line (with trailing newline) ----
    static FString SampleToJsonLine(const FMCTSTrainingSample& S)
    {
        FString L;
        L.Reserve(64 * 1024);

        L += TEXT("{\"phase_id\":");    L.AppendInt(S.PhaseId);
        L += TEXT(",\"player_id\":");   L.AppendInt(S.PlayerId);
        L += TEXT(",\"visit_count\":"); AppendNumber(L, S.VisitCount);

        // Policy target: nonzero entries only
        L += TEXT(",\"policy_size\":"); L.AppendInt(S.PolicyTarget.Num());
        L += TEXT(",\"policy\":[");
        bool bFirst = true;
        for (int32 i = 0; i < S.PolicyTarget.Num(); i++)
        {
            if (S.PolicyTarget[i] <= 0.0f)
                continue;
            if (!bFirst) L += TEXT(",");
            bFirst = false;
            L += TEXT("[");
            L.AppendInt(i);
            L += TEXT(",");
            AppendNumber(L, S.PolicyTarget[i]);
            L += TEXT("]");
        }
        L += TEXT("]");

        // Legal actions: indices only
        L += TEXT(",\"legal_actions\":[");
        bFirst = true;
        for (int32 i = 0; i < S.LegalActionMask.Num(); i++)
        {
            if (!S.LegalActionMask[i])
                continue;
            if (!bFirst) L += TEXT(",");
            bFirst = false;
            L.AppendInt(i);
        }
        L += TEXT("]");

        L += TEXT(",\"value_target\":");    AppendFloatArray(L, S.ValueTarget);
        L += TEXT(",\"node_features\":");   AppendFloatArray(L, S.NodeFeatures);
        L += TEXT(",\"global_features\":"); AppendFloatArray(L, S.GlobalFeatures);

        // Entities: territories that have units only
        L += TEXT(",\"entities\":[");
        bFirst = true;
        for (int32 t = 0; t < S.EntityLists.Num() && t < NUM_TERRITORIES; t++)
        {
            const TArray<FUnitEntity>& Ents = S.EntityLists[t].Entities;
            if (Ents.Num() == 0)
                continue;
            if (!bFirst) L += TEXT(",");
            bFirst = false;
            L += TEXT("[");
            L.AppendInt(t);
            L += TEXT(",[");
            for (int32 e = 0; e < Ents.Num(); e++)
            {
                float F[UNIT_ENTITY_FEATURE_COUNT] = {};
                Ents[e].ToModelFeatures(F);
                if (e > 0) L += TEXT(",");
                L += TEXT("[");
                for (int32 f = 0; f < UNIT_ENTITY_FEATURE_COUNT; f++)
                {
                    if (f > 0) L += TEXT(",");
                    AppendNumber(L, F[f]);
                }
                L += TEXT("]");
            }
            L += TEXT("]]");
        }
        L += TEXT("]}\n");
        return L;
    }

    // ---- Decode 25 model features back into a unit entity ----
    static void DecodeEntityFeatures(const float* F, FUnitEntity& Entity)
    {
        auto Cargo = [F](int32 i, float Scale)
            {
                const float Raw = F[i];
                return Raw > 1e-5f ? FMath::RoundToInt(Raw * Scale) - 1 : -1;
            };

        Entity.UnitType = FMath::RoundToInt(F[0] * 13.0f);
        Entity.OwningPlayer = FMath::RoundToInt(F[1] * 12.0f) - 1;
        Entity.Count = FMath::RoundToInt(F[2] * 20.0f);
        Entity.HitPoints = FMath::RoundToInt(F[3] * 2.0f);
        Entity.MovementRemaining = FMath::RoundToInt(F[4] * 8.0f);
        Entity.SlotId = FMath::RoundToInt(F[5] * 20.0f);
        Entity.bIsScrambled = F[6] > 0.5f;
        Entity.bHasLoadedThisTurn = F[7] > 0.5f;
        Entity.bHasUnloadedThisTurn = F[8] > 0.5f;
        Entity.bIsSubmerged = F[9] > 0.5f;
        Entity.CombatEngagementState = FMath::RoundToInt(F[10] * 2.0f);
        Entity.bIsRetreating = F[11] > 0.5f;
        Entity.CargoUnitTypeA = Cargo(12, 14.0f);
        Entity.CargoUnitOwnerA = Cargo(13, 12.0f);
        Entity.CargoUnitTypeB = Cargo(14, 14.0f);
        Entity.CargoUnitOwnerB = Cargo(15, 12.0f);
        Entity.bIsStrategicBombing = F[16] > 0.5f;
        Entity.bIsEscorting = F[17] > 0.5f;
        Entity.bIsIntercepting = F[18] > 0.5f;
        Entity.bIsConductingSurpriseStrike = F[19] > 0.5f;
        Entity.IsBombarding = FMath::RoundToInt(F[20] * 329.0f);
        Entity.bHasCompletedSurpriseStrike = F[21] > 0.5f;
        Entity.bHasCompletedBombardment = F[22] > 0.5f;
        Entity.bIsParatrooper = F[23] > 0.5f;
        Entity.StartOfTurnTerritory = FMath::RoundToInt(F[24] * 328.0f);
    }

    static bool DecodeEntity(
        const TArray<TSharedPtr<FJsonValue>>& Feat, FUnitEntity& Entity)
    {
        if (Feat.Num() < UNIT_ENTITY_FEATURE_COUNT)
            return false;
        float F[UNIT_ENTITY_FEATURE_COUNT];
        for (int32 i = 0; i < UNIT_ENTITY_FEATURE_COUNT; i++)
            F[i] = (float)Feat[i]->AsNumber();
        DecodeEntityFeatures(F, Entity);
        return true;
    }

    // ---- Binary shard writer ----
    static constexpr uint32 BinaryShardVersion = 1;

    static bool BuildBinaryShard(
        const TArray<const FMCTSTrainingSample*>& Samples, TArray<uint8>& OutFile)
    {
        const int32 N = Samples.Num();
        const int32 NodeSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;

        // ---- Sparse sections, built first so offsets are known ----
        TArray<int32> PolicyOffsets, PolicyIndex, LegalOffsets, LegalIndex;
        TArray<int32> EntityOffsets, EntityTerritory;
        TArray<float> PolicyValue, EntityFeatures;
        PolicyOffsets.Reserve(N + 1);
        LegalOffsets.Reserve(N + 1);
        EntityOffsets.Reserve(N + 1);

        for (const FMCTSTrainingSample* S : Samples)
        {
            PolicyOffsets.Add(PolicyIndex.Num());
            for (int32 i = 0; i < S->PolicyTarget.Num(); i++)
            {
                if (S->PolicyTarget[i] > 0.0f)
                {
                    PolicyIndex.Add(i);
                    PolicyValue.Add(S->PolicyTarget[i]);
                }
            }

            LegalOffsets.Add(LegalIndex.Num());
            for (int32 i = 0; i < S->LegalActionMask.Num(); i++)
                if (S->LegalActionMask[i])
                    LegalIndex.Add(i);

            EntityOffsets.Add(EntityTerritory.Num());
            for (int32 t = 0; t < S->EntityLists.Num() && t < NUM_TERRITORIES; t++)
            {
                for (const FUnitEntity& E : S->EntityLists[t].Entities)
                {
                    float F[UNIT_ENTITY_FEATURE_COUNT] = {};
                    E.ToModelFeatures(F);
                    EntityTerritory.Add(t);
                    EntityFeatures.Append(F, UNIT_ENTITY_FEATURE_COUNT);
                }
            }
        }
        PolicyOffsets.Add(PolicyIndex.Num());
        LegalOffsets.Add(LegalIndex.Num());
        EntityOffsets.Add(EntityTerritory.Num());

        // ---- Payload ----
        TArray<uint8> Raw;
        Raw.Reserve(
            4 + N * 16 +
            N * (NUM_PLAYERS + NodeSize + GLOBAL_FEATURE_COUNT) * 4 +
            (PolicyOffsets.Num() + LegalOffsets.Num() + EntityOffsets.Num()) * 4 +
            (PolicyIndex.Num() * 2 + LegalIndex.Num() + EntityTerritory.Num() +
                EntityFeatures.Num()) * 4);

        auto PutI32 = [&Raw](int32 V) { Raw.Append(reinterpret_cast<const uint8*>(&V), 4); };
        auto PutF32 = [&Raw](float V)
            {
                if (!FMath::IsFinite(V)) V = 0.0f;
                Raw.Append(reinterpret_cast<const uint8*>(&V), 4);
            };
        auto PutI32Array = [&Raw](const TArray<int32>& A)
            { Raw.Append(reinterpret_cast<const uint8*>(A.GetData()), A.Num() * 4); };
        auto PutF32Array = [&PutF32](const TArray<float>& A, int32 Count)
            {
                for (int32 i = 0; i < Count; i++)
                    PutF32(A.IsValidIndex(i) ? A[i] : 0.0f);
            };

        PutI32(N);
        for (const FMCTSTrainingSample* S : Samples) PutI32(S->PhaseId);
        for (const FMCTSTrainingSample* S : Samples) PutI32(S->PlayerId);
        for (const FMCTSTrainingSample* S : Samples) PutI32(S->PolicyTarget.Num());
        for (const FMCTSTrainingSample* S : Samples) PutF32(S->VisitCount);
        for (const FMCTSTrainingSample* S : Samples) PutF32Array(S->ValueTarget, NUM_PLAYERS);
        for (const FMCTSTrainingSample* S : Samples) PutF32Array(S->NodeFeatures, NodeSize);
        for (const FMCTSTrainingSample* S : Samples) PutF32Array(S->GlobalFeatures, GLOBAL_FEATURE_COUNT);
        PutI32Array(PolicyOffsets);
        PutI32Array(PolicyIndex);
        PutF32Array(PolicyValue, PolicyValue.Num());
        PutI32Array(LegalOffsets);
        PutI32Array(LegalIndex);
        PutI32Array(EntityOffsets);
        PutI32Array(EntityTerritory);
        PutF32Array(EntityFeatures, EntityFeatures.Num());

        // ---- Compress ----
        int32 CompressedSize = FCompression::CompressMemoryBound(NAME_Zlib, Raw.Num());
        TArray<uint8> Compressed;
        Compressed.SetNumUninitialized(CompressedSize);
        if (!FCompression::CompressMemory(
            NAME_Zlib, Compressed.GetData(), CompressedSize, Raw.GetData(), Raw.Num()))
        {
            return false;
        }

        // ---- File = header + compressed payload ----
        OutFile.Reset();
        OutFile.Reserve(16 + CompressedSize);
        OutFile.Append(reinterpret_cast<const uint8*>("AAR1"), 4);
        const uint32 Header[3] = {
            BinaryShardVersion, (uint32)Raw.Num(), (uint32)CompressedSize };
        OutFile.Append(reinterpret_cast<const uint8*>(Header), sizeof(Header));
        OutFile.Append(Compressed.GetData(), CompressedSize);
        return true;
    }

    // ---- Binary shard reader ----
    static bool ReadBinaryShard(const FString& Path, TArray<FMCTSTrainingSample>& OutSamples)
    {
        TArray<uint8> File;
        if (!FFileHelper::LoadFileToArray(File, *Path) || File.Num() < 16)
            return false;
        if (FMemory::Memcmp(File.GetData(), "AAR1", 4) != 0)
            return false;

        uint32 Header[3];
        FMemory::Memcpy(Header, File.GetData() + 4, sizeof(Header));
        const int32 RawSize = (int32)Header[1];
        const int32 CompressedSize = (int32)Header[2];
        if (Header[0] != BinaryShardVersion || 16 + CompressedSize > File.Num())
            return false;

        TArray<uint8> Raw;
        Raw.SetNumUninitialized(RawSize);
        if (!FCompression::UncompressMemory(
            NAME_Zlib, Raw.GetData(), RawSize, File.GetData() + 16, CompressedSize))
        {
            return false;
        }

        int64 Cursor = 0;
        bool bOk = true;
        auto Take = [&](int64 Bytes) -> const uint8*
            {
                if (!bOk || Bytes < 0 || Cursor + Bytes > Raw.Num()) { bOk = false; return nullptr; }
                const uint8* Ptr = Raw.GetData() + Cursor;
                Cursor += Bytes;
                return Ptr;
            };
        auto I32 = [](const uint8* Src, int64 i) { int32 V; FMemory::Memcpy(&V, Src + i * 4, 4); return V; };
        auto F32 = [](const uint8* Src, int64 i) { float V; FMemory::Memcpy(&V, Src + i * 4, 4); return V; };

        const uint8* PN = Take(4);
        if (!bOk) return false;
        const int32 N = I32(PN, 0);
        if (N < 0) return false;

        const int32 NodeSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;
        const uint8* Phase = Take((int64)N * 4);
        const uint8* Player = Take((int64)N * 4);
        const uint8* PolSize = Take((int64)N * 4);
        const uint8* Visit = Take((int64)N * 4);
        const uint8* Value = Take((int64)N * NUM_PLAYERS * 4);
        const uint8* Node = Take((int64)N * NodeSize * 4);
        const uint8* Global = Take((int64)N * GLOBAL_FEATURE_COUNT * 4);
        const uint8* PolOff = Take((int64)(N + 1) * 4);
        if (!bOk) return false;
        const int32 NumPolicy = I32(PolOff, N);
        const uint8* PolIdx = Take((int64)NumPolicy * 4);
        const uint8* PolVal = Take((int64)NumPolicy * 4);
        const uint8* LegOff = Take((int64)(N + 1) * 4);
        if (!bOk) return false;
        const int32 NumLegal = I32(LegOff, N);
        const uint8* LegIdx = Take((int64)NumLegal * 4);
        const uint8* EntOff = Take((int64)(N + 1) * 4);
        if (!bOk) return false;
        const int32 NumEntities = I32(EntOff, N);
        const uint8* EntTerr = Take((int64)NumEntities * 4);
        const uint8* EntFeat = Take((int64)NumEntities * UNIT_ENTITY_FEATURE_COUNT * 4);
        if (!bOk) return false;

        OutSamples.Reserve(OutSamples.Num() + N);
        for (int32 i = 0; i < N; i++)
        {
            FMCTSTrainingSample S;
            S.PhaseId = I32(Phase, i);
            S.PlayerId = I32(Player, i);
            S.VisitCount = F32(Visit, i);

            S.ValueTarget.SetNumUninitialized(NUM_PLAYERS);
            FMemory::Memcpy(S.ValueTarget.GetData(), Value + (int64)i * NUM_PLAYERS * 4, NUM_PLAYERS * 4);
            S.NodeFeatures.SetNumUninitialized(NodeSize);
            FMemory::Memcpy(S.NodeFeatures.GetData(), Node + (int64)i * NodeSize * 4, (int64)NodeSize * 4);
            S.GlobalFeatures.SetNumUninitialized(GLOBAL_FEATURE_COUNT);
            FMemory::Memcpy(S.GlobalFeatures.GetData(), Global + (int64)i * GLOBAL_FEATURE_COUNT * 4, GLOBAL_FEATURE_COUNT * 4);

            const int32 Size = FMath::Max(0, I32(PolSize, i));
            S.PolicyTarget.Init(0.0f, Size);
            for (int32 k = I32(PolOff, i); k < I32(PolOff, i + 1); k++)
            {
                const int32 a = I32(PolIdx, k);
                if (S.PolicyTarget.IsValidIndex(a))
                    S.PolicyTarget[a] = F32(PolVal, k);
            }

            S.LegalActionMask.Init(false, Size);
            for (int32 k = I32(LegOff, i); k < I32(LegOff, i + 1); k++)
            {
                const int32 a = I32(LegIdx, k);
                if (S.LegalActionMask.IsValidIndex(a))
                    S.LegalActionMask[a] = true;
            }

            S.EntityLists.SetNum(NUM_TERRITORIES);
            for (int32 k = I32(EntOff, i); k < I32(EntOff, i + 1); k++)
            {
                const int32 t = I32(EntTerr, k);
                if (t < 0 || t >= NUM_TERRITORIES)
                    continue;
                float F[UNIT_ENTITY_FEATURE_COUNT];
                FMemory::Memcpy(F, EntFeat + (int64)k * UNIT_ENTITY_FEATURE_COUNT * 4, sizeof(F));
                FUnitEntity E;
                DecodeEntityFeatures(F, E);
                S.EntityLists[t].Entities.Add(E);
            }

            OutSamples.Add(MoveTemp(S));
        }
        return true;
    }

    // ---- Parse a .jsonl file into JSON objects ----
    static void ReadJsonLines(
        const FString& Path, TArray<TSharedPtr<FJsonObject>>& OutObjects)
    {
        TArray<FString> Lines;
        if (!FFileHelper::LoadFileToStringArray(Lines, *Path))
            return;

        for (const FString& Line : Lines)
        {
            if (Line.TrimStartAndEnd().IsEmpty())
                continue;
            TSharedPtr<FJsonObject> Obj;
            TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Line);
            if (FJsonSerializer::Deserialize(Reader, Obj) && Obj.IsValid())
                OutObjects.Add(Obj);
        }
    }

    // ---- Parse a legacy JSON array file into JSON objects ----
    // Legacy total buffer:   [{"PhaseId":P,"samples":[{...},...]}, ...]
    // Legacy staging file:   [{...sample...}, ...]
    static void ReadLegacyFile(
        const FString& Path, TArray<TSharedPtr<FJsonObject>>& OutObjects)
    {
        FString FileData;
        if (!FFileHelper::LoadFileToString(FileData, *Path))
            return;

        TSharedPtr<FJsonValue> RootValue;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);
        if (!FJsonSerializer::Deserialize(Reader, RootValue) ||
            !RootValue.IsValid() || RootValue->Type != EJson::Array)
            return;

        for (const TSharedPtr<FJsonValue>& Entry : RootValue->AsArray())
        {
            const TSharedPtr<FJsonObject>* ObjPtr = nullptr;
            if (!Entry.IsValid() || !Entry->TryGetObject(ObjPtr))
                continue;

            const TArray<TSharedPtr<FJsonValue>>* SamplesArr = nullptr;
            if ((*ObjPtr)->TryGetArrayField(TEXT("samples"), SamplesArr))
            {
                for (const TSharedPtr<FJsonValue>& SampleVal : *SamplesArr)
                {
                    const TSharedPtr<FJsonObject>* SObjPtr = nullptr;
                    if (SampleVal.IsValid() && SampleVal->TryGetObject(SObjPtr))
                        OutObjects.Add(*SObjPtr);
                }
            }
            else
            {
                OutObjects.Add(*ObjPtr);
            }
        }
    }

    static int32 CountLinesInFile(const FString& Path)
    {
        TArray<FString> Lines;
        if (!FFileHelper::LoadFileToStringArray(Lines, *Path))
            return 0;
        int32 Count = 0;
        for (const FString& Line : Lines)
            if (!Line.TrimStartAndEnd().IsEmpty())
                Count++;
        return Count;
    }

    // ---- Index ----
    static bool SaveIndex(const TArray<FShard>& Shards, int32 NextShardId)
    {
        TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetNumberField(TEXT("next_shard_id"), NextShardId);

        TArray<TSharedPtr<FJsonValue>> ShardArr;
        for (const FShard& S : Shards)
        {
            TSharedPtr<FJsonObject> SObj = MakeShared<FJsonObject>();
            SObj->SetStringField(TEXT("file"), S.File);
            SObj->SetNumberField(TEXT("samples"), S.Samples);
            ShardArr.Add(MakeShared<FJsonValueObject>(SObj));
        }
        Root->SetArrayField(TEXT("shards"), ShardArr);

        FString Output;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
        FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);

        IFileManager::Get().MakeDirectory(*GetReplayDir(), true);
        return FFileHelper::SaveStringToFile(Output, *GetTotalReplayBufferExportPath());
    }

    // Loads the index. If it does not exist yet, creates it; an existing
    // legacy TotalReplayBuffer.json becomes the oldest shard.
    static void LoadIndex(TArray<FShard>& OutShards, int32& OutNextShardId)
    {
        OutShards.Reset();
        OutNextShardId = 1;

        FString FileData;
        if (FFileHelper::LoadFileToString(FileData, *GetTotalReplayBufferExportPath()))
        {
            TSharedPtr<FJsonObject> Root;
            TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);
            if (FJsonSerializer::Deserialize(Reader, Root) && Root.IsValid())
            {
                int32 Next = 1;
                if (Root->TryGetNumberField(TEXT("next_shard_id"), Next))
                    OutNextShardId = FMath::Max(1, Next);

                const TArray<TSharedPtr<FJsonValue>>* ShardArr = nullptr;
                if (Root->TryGetArrayField(TEXT("shards"), ShardArr))
                {
                    for (const TSharedPtr<FJsonValue>& V : *ShardArr)
                    {
                        const TSharedPtr<FJsonObject>* SObj = nullptr;
                        if (!V.IsValid() || !V->TryGetObject(SObj))
                            continue;
                        FShard S;
                        (*SObj)->TryGetStringField(TEXT("file"), S.File);
                        (*SObj)->TryGetNumberField(TEXT("samples"), S.Samples);
                        if (!S.File.IsEmpty())
                            OutShards.Add(S);
                    }
                }
            }
            return;
        }

        // No index yet: adopt the legacy total buffer, if any.
        const FString LegacyPath = GetLegacyTotalPath();
        if (FPaths::FileExists(LegacyPath))
        {
            TArray<TSharedPtr<FJsonObject>> Objects;
            ReadLegacyFile(LegacyPath, Objects);
            FShard Legacy;
            Legacy.File = TEXT("../TotalReplayBuffer.json");
            Legacy.Samples = Objects.Num();
            OutShards.Add(Legacy);
        }
        SaveIndex(OutShards, OutNextShardId);
    }

    static int32 TotalSamples(const TArray<FShard>& Shards)
    {
        int32 Total = 0;
        for (const FShard& S : Shards)
            Total += S.Samples;
        return Total;
    }
}

// ----------------------------------------------------------------
// GetStagingPath — uses CurrentStagingSessionName (append-only .jsonl)
// ----------------------------------------------------------------
FString UAI_ReplayBufferManager::GetStagingPath() const
{
    return FPaths::Combine(
        FPaths::ProjectSavedDir(),
        TEXT("AITraining/Staging"),
        FString::Printf(TEXT("%s.jsonl"), *CurrentStagingSessionName));
}

// ----------------------------------------------------------------
// GetStagingDirectory
// ----------------------------------------------------------------
FString UAI_ReplayBufferManager::GetStagingDirectory() const
{
    return FPaths::Combine(
        FPaths::ProjectSavedDir(),
        TEXT("AITraining/Staging"));
}

// ----------------------------------------------------------------
// GetStagingFileNames
// Returns all staging session names (without extension), including
// sessions that only have a legacy .json staging file.
// ----------------------------------------------------------------
TArray<FString> UAI_ReplayBufferManager::GetStagingFileNames() const
{
    TArray<FString> FileNames;
    const FString Dir = GetStagingDirectory();

    for (const TCHAR* Pattern : { TEXT("*.jsonl"), TEXT("*.json") })
    {
        TArray<FString> FoundFiles;
        IFileManager::Get().FindFiles(FoundFiles, *(Dir / Pattern), true, false);
        for (const FString& File : FoundFiles)
            FileNames.AddUnique(FPaths::GetBaseFilename(File));
    }

    return FileNames;
}

// ----------------------------------------------------------------
// SetStagingSessionName
// Sets session name without loading existing staging file
// ----------------------------------------------------------------
void UAI_ReplayBufferManager::SetStagingSessionName(const FString& SessionName)
{
    CurrentStagingSessionName = SessionName.IsEmpty()
        ? TEXT("Default") : SessionName;
}

// ----------------------------------------------------------------
// LoadStagingSession
// Sets session name and confirms a staging file exists
// ----------------------------------------------------------------
bool UAI_ReplayBufferManager::LoadStagingSession(const FString& SessionName)
{
    SetStagingSessionName(SessionName);
    const FString LegacyPath = FPaths::Combine(
        GetStagingDirectory(),
        FString::Printf(TEXT("%s.json"), *CurrentStagingSessionName));
    return FPaths::FileExists(GetStagingPath()) || FPaths::FileExists(LegacyPath);
}

// ----------------------------------------------------------------
// DeleteStagingFile
// Deletes the staging file(s) for the given session name
// ----------------------------------------------------------------
bool UAI_ReplayBufferManager::DeleteStagingFile(const FString& SessionName)
{
    bool bDeletedAny = false;
    for (const TCHAR* Ext : { TEXT("jsonl"), TEXT("json") })
    {
        const FString Path = FPaths::Combine(
            GetStagingDirectory(),
            FString::Printf(TEXT("%s.%s"), *SessionName, Ext));
        if (FPaths::FileExists(Path))
            bDeletedAny |= IFileManager::Get().Delete(*Path);
    }
    return bDeletedAny;
}

void UAI_ReplayBufferManager::SaveTrainingMetadata(int32 EpisodeId) const
{
    TSharedPtr<FJsonObject> Obj = MakeShared<FJsonObject>();
    Obj->SetNumberField(TEXT("new_samples_since_last_training"), NewSamplesSinceLastTraining);
    Obj->SetNumberField(TEXT("current_episode_id"), EpisodeId);
    Obj->SetStringField(TEXT("current_staging_session_name"), CurrentStagingSessionName);

    FString Output;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);

    const FString Path = GetTrainingMetadataPath();
    const FString Dir = FPaths::GetPath(Path);
    if (!Dir.IsEmpty())
        IFileManager::Get().MakeDirectory(*Dir, true);

    FFileHelper::SaveStringToFile(Output, *Path);
}

void UAI_ReplayBufferManager::SaveTrainingMetadata() const
{
    int32 ExistingEpisodeId = 0;
    const FString Path = GetTrainingMetadataPath();
    FString FileData;
    if (FFileHelper::LoadFileToString(FileData, *Path))
    {
        TSharedPtr<FJsonObject> Obj;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);
        if (FJsonSerializer::Deserialize(Reader, Obj) && Obj.IsValid())
        {
            int32 Loaded = 0;
            if (Obj->TryGetNumberField(TEXT("current_episode_id"), Loaded))
                ExistingEpisodeId = FMath::Max(0, Loaded);
        }
    }

    TSharedPtr<FJsonObject> Obj = MakeShared<FJsonObject>();
    Obj->SetNumberField(TEXT("new_samples_since_last_training"), NewSamplesSinceLastTraining);
    Obj->SetNumberField(TEXT("current_episode_id"), ExistingEpisodeId);
    Obj->SetStringField(TEXT("current_staging_session_name"), CurrentStagingSessionName);

    FString Output;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);

    const FString Dir = FPaths::GetPath(Path);
    if (!Dir.IsEmpty())
        IFileManager::Get().MakeDirectory(*Dir, true);

    FFileHelper::SaveStringToFile(Output, *Path);
}

void UAI_ReplayBufferManager::LoadTrainingMetadata(int32& OutEpisodeId)
{
    const FString Path = GetTrainingMetadataPath();

    FString FileData;
    if (!FFileHelper::LoadFileToString(FileData, *Path))
    {
        NewSamplesSinceLastTraining = 0;
        OutEpisodeId = 0;
        CurrentStagingSessionName = TEXT("Default");
        return;
    }

    TSharedPtr<FJsonObject> Obj;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);

    if (!FJsonSerializer::Deserialize(Reader, Obj) || !Obj.IsValid())
    {
        NewSamplesSinceLastTraining = 0;
        OutEpisodeId = 0;
        CurrentStagingSessionName = TEXT("Default");
        return;
    }

    int32 Loaded = 0;
    if (Obj->TryGetNumberField(TEXT("new_samples_since_last_training"), Loaded))
        NewSamplesSinceLastTraining = FMath::Max(0, Loaded);

    int32 LoadedEpisodeId = 0;
    if (Obj->TryGetNumberField(TEXT("current_episode_id"), LoadedEpisodeId))
        OutEpisodeId = FMath::Max(0, LoadedEpisodeId);
    else
        OutEpisodeId = 0;

    FString LoadedSessionName;
    if (Obj->TryGetStringField(TEXT("current_staging_session_name"), LoadedSessionName))
        CurrentStagingSessionName = LoadedSessionName.IsEmpty()
        ? TEXT("Default") : LoadedSessionName;
    else
        CurrentStagingSessionName = TEXT("Default");
}

void UAI_ReplayBufferManager::LoadTrainingMetadata()
{
    int32 UnusedEpisodeId = 0;
    LoadTrainingMetadata(UnusedEpisodeId);
}

// ----------------------------------------------------------------
// Static helper: FloatArrayToJson (used by legacy SerializeEntityList)
// ----------------------------------------------------------------
static TArray<TSharedPtr<FJsonValue>> FloatArrayToJson(
    const TArray<float>& Arr)
{
    TArray<TSharedPtr<FJsonValue>> Out;
    Out.Reserve(Arr.Num());
    for (float V : Arr)
        Out.Add(MakeShared<FJsonValueNumber>(V));
    return Out;
}

// ----------------------------------------------------------------
// SerializeEntityList — legacy JSON-object writer.
// No longer used for saving (samples are written as JSON lines by
// ReplayStorage::SampleToJsonLine). Kept so the header is unchanged.
// ----------------------------------------------------------------
void UAI_ReplayBufferManager::SerializeEntityList(
    const FMCTSTrainingSample& Sample,
    TSharedPtr<FJsonObject>& SObj) const
{
    TArray<TSharedPtr<FJsonValue>> EntityListJson;
    EntityListJson.Reserve(NUM_TERRITORIES);

    const int32 NumLists = Sample.EntityLists.Num();
    for (int32 t = 0; t < NUM_TERRITORIES; t++)
    {
        TSharedPtr<FJsonObject> TerritoryObj = MakeShared<FJsonObject>();

        if (t < NumLists)
        {
            const FTerritoryEntityList& TList = Sample.EntityLists[t];
            const int32 Count = TList.Entities.Num();
            TerritoryObj->SetNumberField(TEXT("count"), Count);

            TArray<TSharedPtr<FJsonValue>> EntitiesJson;
            EntitiesJson.Reserve(Count);

            for (const FUnitEntity& Entity : TList.Entities)
            {
                float Features[UNIT_ENTITY_FEATURE_COUNT] = {};
                Entity.ToModelFeatures(Features);

                TArray<TSharedPtr<FJsonValue>> FeatJson;
                FeatJson.Reserve(UNIT_ENTITY_FEATURE_COUNT);
                for (int32 f = 0; f < UNIT_ENTITY_FEATURE_COUNT; f++)
                    FeatJson.Add(MakeShared<FJsonValueNumber>(Features[f]));

                EntitiesJson.Add(MakeShared<FJsonValueArray>(FeatJson));
            }

            TerritoryObj->SetArrayField(TEXT("entities"), EntitiesJson);
        }
        else
        {
            TerritoryObj->SetNumberField(TEXT("count"), 0);
            TerritoryObj->SetArrayField(
                TEXT("entities"),
                TArray<TSharedPtr<FJsonValue>>());
        }

        EntityListJson.Add(MakeShared<FJsonValueObject>(TerritoryObj));
    }

    SObj->SetArrayField(TEXT("entity_list"), EntityListJson);
}

// ----------------------------------------------------------------
// DeserializeSample
// Reads one sample from a JSON object. Supports the Stage A line
// format and the legacy format (policy_target / entity_list).
// ----------------------------------------------------------------
void UAI_ReplayBufferManager::DeserializeSample(
    const TSharedPtr<FJsonObject>& SObj,
    FMCTSTrainingSample& Sample) const
{
    const TArray<TSharedPtr<FJsonValue>>* NodeArr = nullptr;
    if (SObj->TryGetArrayField(TEXT("node_features"), NodeArr))
        for (const TSharedPtr<FJsonValue>& V : *NodeArr)
            Sample.NodeFeatures.Add((float)V->AsNumber());

    const TArray<TSharedPtr<FJsonValue>>* GlobalArr = nullptr;
    if (SObj->TryGetArrayField(TEXT("global_features"), GlobalArr))
        for (const TSharedPtr<FJsonValue>& V : *GlobalArr)
            Sample.GlobalFeatures.Add((float)V->AsNumber());

    // ---- Policy target: sparse (Stage A) or dense (legacy) ----
    int32 PolicySize = 0;
    const TArray<TSharedPtr<FJsonValue>>* SparsePolicy = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* DensePolicy = nullptr;
    if (SObj->TryGetNumberField(TEXT("policy_size"), PolicySize) &&
        SObj->TryGetArrayField(TEXT("policy"), SparsePolicy))
    {
        Sample.PolicyTarget.Init(0.0f, FMath::Max(0, PolicySize));
        for (const TSharedPtr<FJsonValue>& Pair : *SparsePolicy)
        {
            const TArray<TSharedPtr<FJsonValue>>* PairArr = nullptr;
            if (!Pair.IsValid() || !Pair->TryGetArray(PairArr) || PairArr->Num() < 2)
                continue;
            const int32 Index = (int32)(*PairArr)[0]->AsNumber();
            if (Sample.PolicyTarget.IsValidIndex(Index))
                Sample.PolicyTarget[Index] = (float)(*PairArr)[1]->AsNumber();
        }
    }
    else if (SObj->TryGetArrayField(TEXT("policy_target"), DensePolicy))
    {
        for (const TSharedPtr<FJsonValue>& V : *DensePolicy)
            Sample.PolicyTarget.Add((float)V->AsNumber());
    }

    const TArray<TSharedPtr<FJsonValue>>* ValueArr = nullptr;
    if (SObj->TryGetArrayField(TEXT("value_target"), ValueArr))
        for (const TSharedPtr<FJsonValue>& V : *ValueArr)
            Sample.ValueTarget.Add((float)V->AsNumber());

    Sample.PhaseId = (int32)SObj->GetNumberField(TEXT("phase_id"));
    Sample.PlayerId = (int32)SObj->GetNumberField(TEXT("player_id"));

    double VisitCount = 0.0;
    if (SObj->TryGetNumberField(TEXT("visit_count"), VisitCount))
        Sample.VisitCount = (float)VisitCount;

    const TArray<TSharedPtr<FJsonValue>>* LegalArr = nullptr;
    if (SObj->TryGetArrayField(TEXT("legal_actions"), LegalArr))
    {
        Sample.LegalActionMask.Init(false, Sample.PolicyTarget.Num());
        for (const TSharedPtr<FJsonValue>& V : *LegalArr)
        {
            const int32 a = (int32)V->AsNumber();
            if (Sample.LegalActionMask.IsValidIndex(a))
                Sample.LegalActionMask[a] = true;
        }
    }

    // ---- Entities: sparse (Stage A) or per-territory (legacy) ----
    Sample.EntityLists.SetNum(NUM_TERRITORIES);

    const TArray<TSharedPtr<FJsonValue>>* SparseEntities = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* EntityListArr = nullptr;
    if (SObj->TryGetArrayField(TEXT("entities"), SparseEntities))
    {
        // [[territory, [[25 features], ...]], ...]
        for (const TSharedPtr<FJsonValue>& TerrVal : *SparseEntities)
        {
            const TArray<TSharedPtr<FJsonValue>>* TerrArr = nullptr;
            if (!TerrVal.IsValid() || !TerrVal->TryGetArray(TerrArr) || TerrArr->Num() < 2)
                continue;
            const int32 t = (int32)(*TerrArr)[0]->AsNumber();
            if (t < 0 || t >= NUM_TERRITORIES)
                continue;

            const TArray<TSharedPtr<FJsonValue>>* EntsArr = nullptr;
            if (!(*TerrArr)[1]->TryGetArray(EntsArr))
                continue;

            for (const TSharedPtr<FJsonValue>& EntityVal : *EntsArr)
            {
                const TArray<TSharedPtr<FJsonValue>>* FeatArr = nullptr;
                if (!EntityVal->TryGetArray(FeatArr))
                    continue;
                FUnitEntity Entity;
                if (ReplayStorage::DecodeEntity(*FeatArr, Entity))
                    Sample.EntityLists[t].Entities.Add(Entity);
            }
        }
    }
    else if (SObj->TryGetArrayField(TEXT("entity_list"), EntityListArr))
    {
        // [{"count":N,"entities":[[25 features], ...]}, ...]
        for (int32 t = 0; t < EntityListArr->Num() && t < NUM_TERRITORIES; t++)
        {
            const TSharedPtr<FJsonObject>* TerritoryObjPtr = nullptr;
            if (!(*EntityListArr)[t]->TryGetObject(TerritoryObjPtr))
                continue;

            const TArray<TSharedPtr<FJsonValue>>* EntitiesArr = nullptr;
            if (!(*TerritoryObjPtr)->TryGetArrayField(TEXT("entities"), EntitiesArr))
                continue;

            for (const TSharedPtr<FJsonValue>& EntityVal : *EntitiesArr)
            {
                const TArray<TSharedPtr<FJsonValue>>* FeatArr = nullptr;
                if (!EntityVal->TryGetArray(FeatArr))
                    continue;
                FUnitEntity Entity;
                if (ReplayStorage::DecodeEntity(*FeatArr, Entity))
                    Sample.EntityLists[t].Entities.Add(Entity);
            }
        }
    }
}

// ----------------------------------------------------------------
// LoadTotalBufferFromDisk
// Reads every shard listed in the index (including a legacy total
// buffer file, if the index adopted one).
// ----------------------------------------------------------------
TArray<FMCTSTrainingSample> UAI_ReplayBufferManager::LoadTotalBufferFromDisk() const
{
    TArray<FMCTSTrainingSample> TotalBuffer;

    TArray<ReplayStorage::FShard> Shards;
    int32 NextShardId = 1;
    ReplayStorage::LoadIndex(Shards, NextShardId);

    for (const ReplayStorage::FShard& Shard : Shards)
    {
        const FString Path = ReplayStorage::GetShardPath(Shard.File);
        if (Shard.File.EndsWith(TEXT(".bin")))
        {
            ReplayStorage::ReadBinaryShard(Path, TotalBuffer);
            continue;
        }
        TArray<TSharedPtr<FJsonObject>> Objects;
        if (Shard.File.EndsWith(TEXT(".jsonl")))
            ReplayStorage::ReadJsonLines(Path, Objects);
        else
            ReplayStorage::ReadLegacyFile(Path, Objects);

        for (const TSharedPtr<FJsonObject>& Obj : Objects)
        {
            FMCTSTrainingSample Sample;
            DeserializeSample(Obj, Sample);
            TotalBuffer.Add(MoveTemp(Sample));
        }
    }

    return TotalBuffer;
}

// Clear samples in memory without saving
void UAI_ReplayBufferManager::ClearBuffer()
{
    Buffer.Empty();
}

// ----------------------------------------------------------------
// SerializeToDisk
// Writes the given samples (a finished game) as ONE new binary shard file,
// adds it to the index, then prunes the oldest shards while the total
// exceeds MaxCapacity. Existing shards are never rewritten.
// ExportPath is kept for signature compatibility (it is the index path).
// ----------------------------------------------------------------
bool UAI_ReplayBufferManager::SerializeToDisk(
    const TArray<FMCTSTrainingSample>& TotalBuffer,
    const FString& ExportPath) const
{
    (void)ExportPath;   // the index location comes from GetTotalReplayBufferExportPath()

    // ---- Collect valid samples ----
    TArray<const FMCTSTrainingSample*> Valid;
    Valid.Reserve(TotalBuffer.Num());
    for (const FMCTSTrainingSample& Sample : TotalBuffer)
        if (IsValidTrainingSample(Sample))
            Valid.Add(&Sample);

    if (Valid.Num() == 0)
        return true;

    TArray<uint8> FileBytes;
    if (!ReplayStorage::BuildBinaryShard(Valid, FileBytes))
        return false;

    // ---- Write the new shard ----
    TArray<ReplayStorage::FShard> Shards;
    int32 NextShardId = 1;
    ReplayStorage::LoadIndex(Shards, NextShardId);

    ReplayStorage::FShard NewShard;
    NewShard.File = FString::Printf(TEXT("shard_%06d.bin"), NextShardId);
    NewShard.Samples = Valid.Num();

    IFileManager::Get().MakeDirectory(*ReplayStorage::GetReplayDir(), true);
    if (!FFileHelper::SaveArrayToFile(FileBytes, *ReplayStorage::GetShardPath(NewShard.File)))
        return false;

    Shards.Add(NewShard);
    NextShardId++;

    // ---- Prune oldest shards while over capacity (keep the newest) ----
    if (MaxCapacity > 0)
    {
        int32 Total = ReplayStorage::TotalSamples(Shards);
        while (Total > MaxCapacity && Shards.Num() > 1)
        {
            IFileManager::Get().Delete(*ReplayStorage::GetShardPath(Shards[0].File));
            Total -= Shards[0].Samples;
            Shards.RemoveAt(0);
        }
    }

    return ReplayStorage::SaveIndex(Shards, NextShardId);
}

// ================================================================
// GetSamplesInMemory
// ================================================================
int32 UAI_ReplayBufferManager::GetSamplesInMemory() const
{
    return Buffer.Num();
}

// ================================================================
// GetSamplesOnDisk — from the index (no file parsing)
// ================================================================
int32 UAI_ReplayBufferManager::GetSamplesOnDisk() const
{
    TArray<ReplayStorage::FShard> Shards;
    int32 NextShardId = 1;
    ReplayStorage::LoadIndex(Shards, NextShardId);
    return ReplayStorage::TotalSamples(Shards);
}

// ================================================================
// GetSamplesInStagingFile — current session (new + legacy files)
// ================================================================
int32 UAI_ReplayBufferManager::GetSamplesInStagingFile() const
{
    int32 Count = ReplayStorage::CountLinesInFile(GetStagingPath());

    const FString LegacyPath = FPaths::Combine(
        GetStagingDirectory(),
        FString::Printf(TEXT("%s.json"), *CurrentStagingSessionName));
    if (FPaths::FileExists(LegacyPath))
    {
        TArray<TSharedPtr<FJsonObject>> Objects;
        ReplayStorage::ReadLegacyFile(LegacyPath, Objects);
        Count += Objects.Num();
    }
    return Count;
}

// ================================================================
// FlushPartialToDisk
// Appends the in-memory samples to the session's staging file with
// placeholder value targets. Nothing already on disk is re-read or
// rewritten.
// ================================================================
bool UAI_ReplayBufferManager::FlushPartialToDisk()
{
    if (Buffer.Num() == 0)
        return true;

    FString Content;
    for (const FMCTSTrainingSample& S : Buffer)
    {
        FMCTSTrainingSample Copy = S;
        Copy.ValueTarget.Init(0.0f, NUM_PLAYERS);
        Content += ReplayStorage::SampleToJsonLine(Copy);
    }

    const FString StagingPath = GetStagingPath();
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(StagingPath), true);

    const bool bSuccess = FFileHelper::SaveStringToFile(
        Content, *StagingPath,
        FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM,
        &IFileManager::Get(), FILEWRITE_Append);

    if (bSuccess)
        Buffer.Empty();

    return bSuccess;
}

// ================================================================
// ApplyOutcomeValuesToEpisode
// Reads the session's staging samples, applies the final outcome,
// writes them as one new shard, and deletes the staging file(s).
// ================================================================
bool UAI_ReplayBufferManager::ApplyOutcomeValuesToEpisode(
    const TArray<float>& OutcomeValues)
{
    if (OutcomeValues.Num() != NUM_PLAYERS)
    {
        return false;
    }

    const FString StagingPath = GetStagingPath();
    const FString LegacyPath = FPaths::Combine(
        GetStagingDirectory(),
        FString::Printf(TEXT("%s.json"), *CurrentStagingSessionName));

    const bool bHasStaging = FPaths::FileExists(StagingPath);
    const bool bHasLegacy = FPaths::FileExists(LegacyPath);

    // No staging file — all samples still in memory.
    // ExportReplayBuffer handles them normally.
    if (!bHasStaging && !bHasLegacy)
        return true;

    // ---- Load staging samples (legacy first: they are older) ----
    TArray<TSharedPtr<FJsonObject>> Objects;
    if (bHasLegacy)
        ReplayStorage::ReadLegacyFile(LegacyPath, Objects);
    if (bHasStaging)
        ReplayStorage::ReadJsonLines(StagingPath, Objects);

    TArray<FMCTSTrainingSample> StagingBuffer;
    StagingBuffer.Reserve(Objects.Num());
    for (const TSharedPtr<FJsonObject>& Obj : Objects)
    {
        FMCTSTrainingSample Sample;
        DeserializeSample(Obj, Sample);
        StagingBuffer.Add(MoveTemp(Sample));
    }

    // ---- Apply actual outcome values ----
    for (FMCTSTrainingSample& Sample : StagingBuffer)
    {
        Sample.ValueTarget.SetNum(NUM_PLAYERS);
        for (int32 p = 0; p < NUM_PLAYERS; p++)
            Sample.ValueTarget[p] = OutcomeValues[p];
    }

    // ---- Drop invalid samples so the counter matches what is saved ----
    StagingBuffer.RemoveAll([this](const FMCTSTrainingSample& S)
        { return !IsValidTrainingSample(S); });

    // ---- Write as a new shard ----
    if (StagingBuffer.Num() > 0)
    {
        if (!SerializeToDisk(StagingBuffer, GetTotalReplayBufferExportPath()))
            return false;   // keep the staging file so nothing is lost

        NewSamplesSinceLastTraining += StagingBuffer.Num();
        SaveTrainingMetadata();
    }

    // ---- Delete staging file(s) ----
    if (bHasStaging) IFileManager::Get().Delete(*StagingPath);
    if (bHasLegacy)  IFileManager::Get().Delete(*LegacyPath);

    return true;
}

UAI_ReplayBufferManager& UAI_ReplayBufferManager::Get()
{
    static UAI_ReplayBufferManager Instance;

    if (!Instance.bMetadataLoaded)
    {
        Instance.LoadTrainingMetadata();
        Instance.bMetadataLoaded = true;
    }

    return Instance;
}

// ================================================================
// ExportReplayBuffer
// Applies the final outcome to the in-memory samples and writes them
// as one new shard. Existing shards are not loaded or rewritten.
// ================================================================
bool UAI_ReplayBufferManager::ExportReplayBuffer(
    const TArray<float>& FinalOutcomeValues)
{
    const int32 ExpectedPlayers = FinalOutcomeValues.Num();
    if (ExpectedPlayers <= 0)
    {
        ensureMsgf(false,
            TEXT("ExportReplayBuffer: invalid FinalOutcomeValues size"));
        return false;
    }

    // ---- Apply final game rewards to in-memory samples ----
    for (FMCTSTrainingSample& Sample : Buffer)
    {
        Sample.ValueTarget.SetNumZeroed(ExpectedPlayers);
        for (int32 PlayerIdx = 0; PlayerIdx < ExpectedPlayers; ++PlayerIdx)
            Sample.ValueTarget[PlayerIdx] = FinalOutcomeValues[PlayerIdx];
    }

    // ---- Drop invalid samples so the counter matches what is saved ----
    Buffer.RemoveAll([this](const FMCTSTrainingSample& S)
        { return !IsValidTrainingSample(S); });

    if (Buffer.Num() == 0)
        return true;

    // ---- Write as a new shard ----
    const bool bSuccess =
        SerializeToDisk(Buffer, GetTotalReplayBufferExportPath());

    if (bSuccess)
    {
        NewSamplesSinceLastTraining += Buffer.Num();
        SaveTrainingMetadata();
        Buffer.Empty();
    }

    return bSuccess;
}

void UAI_ReplayBufferManager::StoreSelfPlaySample(const FMCTSTree& Tree)
{

    if (!Tree.Nodes.IsValidIndex(Tree.RootIndex))
        return;
    const FMCTSNode& Root = Tree.Nodes[Tree.RootIndex];
    if (Root.ChildIndices.Num() == 0 || Root.VisitCount <= 0)
        return;
    FMCTSTrainingSample Sample;
    // ------------------------------------------------------------
    // GRAPH INPUT
    // ------------------------------------------------------------
    Sample.NodeFeatures = Root.NodeFeatures;
    Sample.GlobalFeatures = Root.GlobalFeatures;
    Sample.EntityLists = Root.EntityLists;
    if (Sample.EntityLists.Num() != NUM_TERRITORIES)
        Sample.EntityLists.SetNum(NUM_TERRITORIES);
    // ------------------------------------------------------------
    // CONTEXT
    // ------------------------------------------------------------
    Sample.PhaseId = Root.PhaseId;
    Sample.PlayerId = Root.PlayerId;
    Sample.GameId = Root.GameId;
    Sample.MoveIndex = Root.MoveIndex;
    // ------------------------------------------------------------
    // LEGAL ACTIONS (used by training to mask the policy loss)
    // ------------------------------------------------------------
    Sample.LegalActionMask = Root.LegalActionMask;
    // ------------------------------------------------------------
    // POLICY TARGET
    // ------------------------------------------------------------
    const int32 ActionSpaceSize = Root.LegalActionMask.Num();
    Sample.PolicyTarget.Init(0.0f, ActionSpaceSize);
    float TotalVisits = 0.0f;
    for (int32 Action = 0; Action < ActionSpaceSize; ++Action)
    {
        const bool bLegal =
            Root.LegalActionMask.IsValidIndex(Action) &&
            Root.LegalActionMask[Action];
        if (!bLegal)
            continue;
        const int32 Visits =
            (Action < Root.EdgeVisitCount.Num())
            ? Root.EdgeVisitCount[Action]
            : 0;
        const float V = static_cast<float>(FMath::Max(Visits, 0));
        Sample.PolicyTarget[Action] = V;
        TotalVisits += V;
    }
    if (TotalVisits > 0.0f)
    {
        const float InvTotal = 1.0f / TotalVisits;
        for (float& P : Sample.PolicyTarget)
            P *= InvTotal;
    }
    // ------------------------------------------------------------
    // VALUE TARGET
    // Terminal nodes use outcome values from global features.
    // Non-terminal nodes use MCTSValuePerPlayer from inference.
    // (Replaced by the real game outcome when the game ends.)
    // ------------------------------------------------------------
    Sample.ValueTarget.SetNum(NUM_PLAYERS);
    if (Root.bIsTerminal)
    {
        for (int32 p = 0; p < NUM_PLAYERS; ++p)
        {
            const int32 OutcomeIndex = GLOBAL_OUTCOME_VALUES + p;
            Sample.ValueTarget[p] =
                Root.GlobalFeatures.IsValidIndex(OutcomeIndex)
                ? Root.GlobalFeatures[OutcomeIndex]
                : 0.0f;
        }
    }
    else
    {
        for (int32 p = 0; p < NUM_PLAYERS; ++p)
        {
            Sample.ValueTarget[p] =
                Root.MCTSValuePerPlayer.IsValidIndex(p)
                ? Root.MCTSValuePerPlayer[p]
                : 0.0f;
        }
    }
    // ------------------------------------------------------------
    // TRAINING SIGNAL
    // ------------------------------------------------------------
    Sample.Reward = 0.0f;
    Sample.VisitCount = static_cast<float>(Root.VisitCount);
    if (!IsValidTrainingSample(Sample))
        return;
    Buffer.Add(Sample);
}

void UAI_ReplayBufferManager::AddSample(const FMCTSTrainingSample& Sample)
{
    // ------------------------------------------------------------
    // VALIDATION: GRAPH TRANSFORMER INPUT CONTRACT
    // ------------------------------------------------------------

    const int32 NodeFeatureSizeExpected =
        NUM_TERRITORIES * NODE_FEATURE_COUNT;

    if (Sample.NodeFeatures.Num() != NodeFeatureSizeExpected)
    {
        return;
    }

    if (Sample.GlobalFeatures.Num() != GLOBAL_FEATURE_COUNT)
    {
        return;
    }

    if (Sample.ValueTarget.Num() != NUM_PLAYERS)
    {
        return;
    }

    if (Sample.PhaseId < 0)
    {
        return;
    }

    // ------------------------------------------------------------
    // POLICY HEAD SIZE VALIDATION (PHASE-DRIVEN)
    // ------------------------------------------------------------

    const int32 ExpectedPolicySize =
        GetPolicySizeForPhase(static_cast<EPhaseId>(Sample.PhaseId));

    if (ExpectedPolicySize <= 0)
    {
        return;
    }

    if (Sample.PolicyTarget.Num() != ExpectedPolicySize)
    {
        return;
    }

    // ------------------------------------------------------------
    // BUFFER INSERTION (NO TRANSFORMATION)
    // ------------------------------------------------------------
    if (!IsValidTrainingSample(Sample)) return;
    Buffer.Add(Sample);
}

// Kept for compatibility. On-disk capacity is now enforced per shard
// inside SerializeToDisk.
void UAI_ReplayBufferManager::EnforceCapacity(TArray<FMCTSTrainingSample>& InOutBuffer)
{
    if (InOutBuffer.Num() <= MaxCapacity)
    {
        return;
    }

    const int32 Overflow = InOutBuffer.Num() - MaxCapacity;

    InOutBuffer.RemoveAt(
        0,
        Overflow,
        EAllowShrinking::No);
}

void UAI_ReplayBufferManager::SetMaxCapacity(int32 InMaxCapacity)
{
    MaxCapacity = InMaxCapacity;
}

const TArray<FMCTSTrainingSample>& UAI_ReplayBufferManager::GetBuffer() const
{
    return Buffer;
}

int32 UAI_ReplayBufferManager::GetSampleCount() const
{
    return Buffer.Num();
}

void UAI_ReplayBufferManager::Clear()
{
    Buffer.Empty();
}

void UAI_ReplayBufferManager::BeginGameSession()
{
    Buffer.Empty();
}

void UAI_ReplayBufferManager::ApplyFinalGameRewardsToReplayBuffer(const TArray<float>& FinalOutcomeValues)
{
    const int32 NumPlayers = FinalOutcomeValues.Num();

    // -----------------------------
    // HARD GUARD: INVALID GAME OUTCOME
    // -----------------------------
    if (NumPlayers <= 0)
    {
        return;
    }

    // -----------------------------
    // CONSISTENCY VALIDATION
    // (Ensures alignment with model.py value head ordering)
    // -----------------------------
    auto ValidateOrAbort = [&](const FMCTSTrainingSample& Sample)
        {
            return Sample.ValueTarget.Num() == 0 ||
                Sample.ValueTarget.Num() == NumPlayers;
        };

    auto ApplyToBuffer = [&](TArray<FMCTSTrainingSample>& InBuffer)
        {
            for (FMCTSTrainingSample& Sample : InBuffer)
            {
                // -----------------------------
                // VALUE HEAD SHAPE GUARANTEE
                // -----------------------------
                if (!ValidateOrAbort(Sample))
                {
                    // Prevent silent corruption of training distribution
                    continue;
                }

                if (Sample.ValueTarget.Num() != NumPlayers)
                {
                    Sample.ValueTarget.SetNumZeroed(NumPlayers);
                }

                // -----------------------------
                // STRICT PLAYER-ID INDEX BINDING
                // (No semantic transformation allowed)
                // -----------------------------
                for (int32 PlayerIdx = 0; PlayerIdx < NumPlayers; ++PlayerIdx)
                {
                    const float OutcomeValue =
                        FinalOutcomeValues.IsValidIndex(PlayerIdx)
                        ? FinalOutcomeValues[PlayerIdx]
                        : 0.0f;

                    Sample.ValueTarget[PlayerIdx] = OutcomeValue;
                }
            }
        };

    // -----------------------------
    // APPLY TO EPISODE BUFFER (AUTHORITATIVE SOURCE)
    // -----------------------------
    ApplyToBuffer(Buffer);

    // -----------------------------
    // OPTIONAL SAFETY PASS:
    // ENSURE BUFFER REMAINS VALUE-CONSISTENT AFTER MUTATION
    // -----------------------------
    for (FMCTSTrainingSample& Sample : Buffer)
    {
        if (Sample.ValueTarget.Num() != NumPlayers)
        {
            Sample.ValueTarget.SetNumZeroed(NumPlayers);
        }
    }
}

bool UAI_ReplayBufferManager::ValidateAndNormalizeReplayExport(
    TArray<FMCTSTrainingSample>& InOutTotalBuffer,
    const TArray<FMCTSTrainingSample>& InMemoryBuffer,
    const TArray<float>& FinalOutcomeValues)
{
    if (FinalOutcomeValues.Num() == 0)
    {
        return false;
    }

    // -----------------------------
    // VALIDATE VALUE HEAD ALIGNMENT
    // -----------------------------
    for (const FMCTSTrainingSample& Sample : InMemoryBuffer)
    {
        if (Sample.PlayerId < 0 || Sample.PlayerId >= FinalOutcomeValues.Num())
        {
            return false;
        }

        if (Sample.ValueTarget.Num() != FinalOutcomeValues.Num())
        {
            return false;
        }
    }

    // -----------------------------
    // APPLY REWARD PROPAGATION
    // -----------------------------
    TArray<FMCTSTrainingSample> NormalizedBuffer;
    NormalizedBuffer.Reserve(InMemoryBuffer.Num());

    for (FMCTSTrainingSample Sample : InMemoryBuffer)
    {
        const int32 PlayerIdx = Sample.PlayerId;

        if (PlayerIdx < 0 || PlayerIdx >= FinalOutcomeValues.Num())
        {
            continue;
        }

        // enforce strict model.py value head ordering
        Sample.ValueTarget.Empty();
        Sample.ValueTarget.SetNum(FinalOutcomeValues.Num());

        for (int32 i = 0; i < FinalOutcomeValues.Num(); ++i)
        {
            Sample.ValueTarget[i] = FinalOutcomeValues[i];
        }

        NormalizedBuffer.Add(MoveTemp(Sample));
    }

    // -----------------------------
    // PHASE-ID CONSISTENCY SANITY CHECK
    // -----------------------------
    TSet<int32> PhaseSet;

    for (const FMCTSTrainingSample& Sample : NormalizedBuffer)
    {
        PhaseSet.Add(Sample.PhaseId);
    }

    if (PhaseSet.Num() == 0)
    {
        return false;
    }

    // -----------------------------
    // POLICY TARGET VALIDATION (LEGAL MASK ENFORCEMENT POINT)
    // -----------------------------
    for (const FMCTSTrainingSample& Sample : NormalizedBuffer)
    {
        const int32 ActionCount = Sample.PolicyTarget.Num();

        if (ActionCount <= 0)
        {
            return false;
        }

        // Ensure no invalid probabilities exist (NaN / negative)
        for (int32 i = 0; i < ActionCount; ++i)
        {
            if (!FMath::IsFinite(Sample.PolicyTarget[i]))
            {
                return false;
            }

            if (Sample.PolicyTarget[i] < 0.0f)
            {
                return false;
            }
        }

        // NOTE:
        // Masking must already have been applied BEFORE buffer insertion.
        // We validate normalization correctness here (sum sanity only).
        float Sum = 0.0f;
        for (float V : Sample.PolicyTarget)
        {
            Sum += V;
        }

        if (Sum <= 0.0f)
        {
            return false;
        }
    }

    // -----------------------------
    // MERGE INTO TOTAL BUFFER (ORDER PRESERVING)
    // -----------------------------
    InOutTotalBuffer.Append(NormalizedBuffer);

    // -----------------------------
    // FINAL DEDUP + STABILITY PASS
    // -----------------------------
    TMap<int32, int32> PhaseCount;

    TArray<FMCTSTrainingSample> FinalBuffer;
    FinalBuffer.Reserve(InOutTotalBuffer.Num());

    for (const FMCTSTrainingSample& Sample : InOutTotalBuffer)
    {
        int32& Count = PhaseCount.FindOrAdd(Sample.PhaseId);

        if (Count < 100000)
        {
            FinalBuffer.Add(Sample);
            Count++;
        }
    }

    InOutTotalBuffer = MoveTemp(FinalBuffer);

    return true;
}

bool UAI_ReplayBufferManager::IsValidTrainingSample(const FMCTSTrainingSample& Sample) const
{
    // -----------------------------
    // CORE FEATURE VALIDATION
    // -----------------------------
    const int32 ExpectedNodeSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;

    if (Sample.NodeFeatures.Num() != ExpectedNodeSize)
    {
        return false;
    }

    if (Sample.GlobalFeatures.Num() != GLOBAL_FEATURE_COUNT)
    {
        return false;
    }

    // -----------------------------
    // VALUE HEAD VALIDATION (STRICT PLAYER INDEX CONTRACT)
    // -----------------------------
    if (Sample.ValueTarget.Num() != NUM_PLAYERS)
    {
        return false;
    }

    if (Sample.PlayerId < 0 || Sample.PlayerId >= NUM_PLAYERS)
    {
        return false;
    }

    // -----------------------------
    // POLICY HEAD VALIDATION (PHASE → ACTION SPACE CONTRACT)
    // -----------------------------
    const int32 ExpectedPolicySize =
        GetPolicySizeForPhase(static_cast<EPhaseId>(Sample.PhaseId));

    if (ExpectedPolicySize <= 0)
    {
        return false;
    }

    if (Sample.PolicyTarget.Num() != ExpectedPolicySize)
    {
        return false;
    }

    // -----------------------------
    // SANITY: NO INVALID NUMERIC VALUES
    // -----------------------------
    for (float V : Sample.PolicyTarget)
    {
        if (!FMath::IsFinite(V) || V < 0.0f)
        {
            return false;
        }
    }

    return true;
}

// ================================================================
// GetDiskSampleCount — from the index (no file parsing)
// ================================================================
int32 UAI_ReplayBufferManager::GetDiskSampleCount() const
{
    return GetSamplesOnDisk();
}

int32 UAI_ReplayBufferManager::GetNewSamplesSinceLastTraining() const
{
    return NewSamplesSinceLastTraining;
}

void UAI_ReplayBufferManager::ResetNewSampleCounter()
{
    NewSamplesSinceLastTraining = 0;
    SaveTrainingMetadata();
}