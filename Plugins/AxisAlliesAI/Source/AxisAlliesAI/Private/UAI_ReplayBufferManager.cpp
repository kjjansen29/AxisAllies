#include "UAI_ReplayBufferManager.h"

// ----------------------------------------------------------------
// GetStagingPath — uses CurrentStagingSessionName
// ----------------------------------------------------------------
FString UAI_ReplayBufferManager::GetStagingPath() const
{
    return FPaths::Combine(
        FPaths::ProjectSavedDir(),
        TEXT("AITraining/Staging"),
        FString::Printf(TEXT("%s.json"), *CurrentStagingSessionName));
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
// Returns all staging file names without extension
// ----------------------------------------------------------------
TArray<FString> UAI_ReplayBufferManager::GetStagingFileNames() const
{
    TArray<FString> FileNames;
    const FString Dir = GetStagingDirectory();

    TArray<FString> FoundFiles;
    IFileManager::Get().FindFiles(FoundFiles, *(Dir / TEXT("*.json")), true, false);

    for (const FString& File : FoundFiles)
        FileNames.Add(FPaths::GetBaseFilename(File));

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
// Sets session name and confirms staging file exists
// ----------------------------------------------------------------
bool UAI_ReplayBufferManager::LoadStagingSession(const FString& SessionName)
{
    SetStagingSessionName(SessionName);
    return FPaths::FileExists(GetStagingPath());
}

// ----------------------------------------------------------------
// DeleteStagingFile
// Deletes the staging file for the given session name
// ----------------------------------------------------------------
bool UAI_ReplayBufferManager::DeleteStagingFile(const FString& SessionName)
{
    const FString Path = FPaths::Combine(
        GetStagingDirectory(),
        FString::Printf(TEXT("%s.json"), *SessionName));

    if (!FPaths::FileExists(Path))
        return false;

    return IFileManager::Get().Delete(*Path);
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
// Static helper: FloatArrayToJson
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
// Instance method: SerializeEntityList
// Writes entity_list field to SObj.
// Mirrors ToModelFeatures() exactly — 25 features.
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
// Instance method: DeserializeSample
// Reads one sample from a JSON object.
// Mirrors ToModelFeatures() exactly — 25 features.
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

    const TArray<TSharedPtr<FJsonValue>>* PolicyArr = nullptr;
    if (SObj->TryGetArrayField(TEXT("policy_target"), PolicyArr))
        for (const TSharedPtr<FJsonValue>& V : *PolicyArr)
            Sample.PolicyTarget.Add((float)V->AsNumber());

    const TArray<TSharedPtr<FJsonValue>>* ValueArr = nullptr;
    if (SObj->TryGetArrayField(TEXT("value_target"), ValueArr))
        for (const TSharedPtr<FJsonValue>& V : *ValueArr)
            Sample.ValueTarget.Add((float)V->AsNumber());

    Sample.PhaseId = (int32)SObj->GetNumberField(TEXT("phase_id"));
    Sample.PlayerId = (int32)SObj->GetNumberField(TEXT("player_id"));

    const TArray<TSharedPtr<FJsonValue>>* EntityListArr = nullptr;
    if (SObj->TryGetArrayField(TEXT("entity_list"), EntityListArr))
    {
        Sample.EntityLists.SetNum(NUM_TERRITORIES);
        for (int32 t = 0;
            t < EntityListArr->Num() && t < NUM_TERRITORIES; t++)
        {
            const TSharedPtr<FJsonObject>* TerritoryObjPtr = nullptr;
            if (!(*EntityListArr)[t]->TryGetObject(TerritoryObjPtr))
                continue;

            FTerritoryEntityList& TerritoryList = Sample.EntityLists[t];
            TerritoryList.Reset();

            const TArray<TSharedPtr<FJsonValue>>* EntitiesArr = nullptr;
            if (!(*TerritoryObjPtr)->TryGetArrayField(
                TEXT("entities"), EntitiesArr))
                continue;

            for (const TSharedPtr<FJsonValue>& EntityVal : *EntitiesArr)
            {
                const TArray<TSharedPtr<FJsonValue>>* FeatArr = nullptr;
                if (!EntityVal->TryGetArray(FeatArr))
                    continue;
                if (FeatArr->Num() < UNIT_ENTITY_FEATURE_COUNT)
                    continue;

                FUnitEntity Entity;

                // [0]  UnitType / 13
                Entity.UnitType =
                    FMath::RoundToInt((*FeatArr)[0]->AsNumber() * 13.0f);
                // [1]  (OwningPlayer+1) / 12
                Entity.OwningPlayer =
                    FMath::RoundToInt((*FeatArr)[1]->AsNumber() * 12.0f) - 1;
                // [2]  min(Count,20) / 20
                Entity.Count =
                    FMath::RoundToInt((*FeatArr)[2]->AsNumber() * 20.0f);
                // [3]  HitPoints / 2
                Entity.HitPoints =
                    FMath::RoundToInt((*FeatArr)[3]->AsNumber() * 2.0f);
                // [4]  min(MovementRemaining,8) / 8
                Entity.MovementRemaining =
                    FMath::RoundToInt((*FeatArr)[4]->AsNumber() * 8.0f);
                // [5]  SlotId / 20
                Entity.SlotId =
                    FMath::RoundToInt((*FeatArr)[5]->AsNumber() * 20.0f);
                // [6]  bIsScrambled
                Entity.bIsScrambled =
                    (*FeatArr)[6]->AsNumber() > 0.5f;
                // [7]  bHasLoadedThisTurn
                Entity.bHasLoadedThisTurn =
                    (*FeatArr)[7]->AsNumber() > 0.5f;
                // [8]  bHasUnloadedThisTurn
                Entity.bHasUnloadedThisTurn =
                    (*FeatArr)[8]->AsNumber() > 0.5f;
                // [9]  bIsSubmerged
                Entity.bIsSubmerged =
                    (*FeatArr)[9]->AsNumber() > 0.5f;
                // [10] CombatEngagementState / 2
                Entity.CombatEngagementState =
                    FMath::RoundToInt((*FeatArr)[10]->AsNumber() * 2.0f);
                // [11] bIsRetreating
                Entity.bIsRetreating =
                    (*FeatArr)[11]->AsNumber() > 0.5f;
                // [12] (CargoUnitTypeA+1)/14; 0 if empty
                {
                    const float Raw = (float)(*FeatArr)[12]->AsNumber();
                    Entity.CargoUnitTypeA = Raw > 1e-5f
                        ? FMath::RoundToInt(Raw * 14.0f) - 1 : -1;
                }
                // [13] (CargoUnitOwnerA+1)/12; 0 if empty
                {
                    const float Raw = (float)(*FeatArr)[13]->AsNumber();
                    Entity.CargoUnitOwnerA = Raw > 1e-5f
                        ? FMath::RoundToInt(Raw * 12.0f) - 1 : -1;
                }
                // [14] (CargoUnitTypeB+1)/14; 0 if empty
                {
                    const float Raw = (float)(*FeatArr)[14]->AsNumber();
                    Entity.CargoUnitTypeB = Raw > 1e-5f
                        ? FMath::RoundToInt(Raw * 14.0f) - 1 : -1;
                }
                // [15] (CargoUnitOwnerB+1)/12; 0 if empty
                {
                    const float Raw = (float)(*FeatArr)[15]->AsNumber();
                    Entity.CargoUnitOwnerB = Raw > 1e-5f
                        ? FMath::RoundToInt(Raw * 12.0f) - 1 : -1;
                }
                // [16] bIsStrategicBombing
                Entity.bIsStrategicBombing =
                    (*FeatArr)[16]->AsNumber() > 0.5f;
                // [17] bIsEscorting
                Entity.bIsEscorting =
                    (*FeatArr)[17]->AsNumber() > 0.5f;
                // [18] bIsIntercepting
                Entity.bIsIntercepting =
                    (*FeatArr)[18]->AsNumber() > 0.5f;
                // [19] bIsConductingSurpriseStrike
                Entity.bIsConductingSurpriseStrike =
                    (*FeatArr)[19]->AsNumber() > 0.5f;
                // [20] IsBombarding / 329
                Entity.IsBombarding =
                    FMath::RoundToInt((*FeatArr)[20]->AsNumber() * 329.0f);
                // [21] bHasCompletedSurpriseStrike
                Entity.bHasCompletedSurpriseStrike =
                    (*FeatArr)[21]->AsNumber() > 0.5f;
                // [22] bHasCompletedBombardment
                Entity.bHasCompletedBombardment =
                    (*FeatArr)[22]->AsNumber() > 0.5f;
                // [23] bIsParatrooper
                Entity.bIsParatrooper =
                    (*FeatArr)[23]->AsNumber() > 0.5f;
                // [24] StartOfTurnTerritory / 328
                Entity.StartOfTurnTerritory =
                    FMath::RoundToInt((*FeatArr)[24]->AsNumber() * 328.0f);

                TerritoryList.Entities.Add(Entity);
            }
        }
    }
    else
    {
        Sample.EntityLists.SetNum(NUM_TERRITORIES);
    }
}

// ----------------------------------------------------------------
// Instance method: LoadTotalBufferFromDisk
// ----------------------------------------------------------------
TArray<FMCTSTrainingSample> UAI_ReplayBufferManager::LoadTotalBufferFromDisk() const
{
    TArray<FMCTSTrainingSample> TotalBuffer;
    const FString ExportPath = GetTotalReplayBufferExportPath();

    FString FileData;
    if (!FFileHelper::LoadFileToString(FileData, *ExportPath))
        return TotalBuffer;

    TSharedPtr<FJsonValue> RootValue;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);
    if (!FJsonSerializer::Deserialize(Reader, RootValue) ||
        !RootValue.IsValid() ||
        RootValue->Type != EJson::Array)
        return TotalBuffer;

    for (const TSharedPtr<FJsonValue>& PhaseEntry : RootValue->AsArray())
    {
        const TSharedPtr<FJsonObject>* PhaseObjPtr = nullptr;
        if (!PhaseEntry.IsValid() || !PhaseEntry->TryGetObject(PhaseObjPtr))
            continue;
        const TArray<TSharedPtr<FJsonValue>>* SamplesArr = nullptr;
        if (!(*PhaseObjPtr)->TryGetArrayField(TEXT("samples"), SamplesArr))
            continue;
        for (const TSharedPtr<FJsonValue>& SampleVal : *SamplesArr)
        {
            const TSharedPtr<FJsonObject>* SObjPtr = nullptr;
            if (!SampleVal.IsValid() || !SampleVal->TryGetObject(SObjPtr))
                continue;
            FMCTSTrainingSample Sample;
            DeserializeSample(*SObjPtr, Sample);
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
// Instance method: SerializeToDisk
// Writes TotalBuffer to disk grouped by PhaseId.
// ----------------------------------------------------------------
bool UAI_ReplayBufferManager::SerializeToDisk(
    const TArray<FMCTSTrainingSample>& TotalBuffer,
    const FString& ExportPath) const
{
    TMap<int32, TArray<const FMCTSTrainingSample*>> PhaseGroups;
    for (const FMCTSTrainingSample& Sample : TotalBuffer)
        PhaseGroups.FindOrAdd(Sample.PhaseId).Add(&Sample);

    TArray<TSharedPtr<FJsonValue>> RootArray;
    for (auto& It : PhaseGroups)
    {
        const int32 PhaseId = It.Key;
        const TArray<const FMCTSTrainingSample*>& Samples = It.Value;

        TArray<TSharedPtr<FJsonValue>> SamplesArray;
        SamplesArray.Reserve(Samples.Num());

        for (const FMCTSTrainingSample* Sample : Samples)
        {
            if (!IsValidTrainingSample(*Sample))
                continue;

            TSharedPtr<FJsonObject> SObj = MakeShared<FJsonObject>();
            SObj->SetArrayField(TEXT("node_features"),
                FloatArrayToJson(Sample->NodeFeatures));
            SObj->SetArrayField(TEXT("global_features"),
                FloatArrayToJson(Sample->GlobalFeatures));
            SObj->SetArrayField(TEXT("policy_target"),
                FloatArrayToJson(Sample->PolicyTarget));
            SObj->SetArrayField(TEXT("value_target"),
                FloatArrayToJson(Sample->ValueTarget));
            SObj->SetNumberField(TEXT("phase_id"), Sample->PhaseId);
            SObj->SetNumberField(TEXT("player_id"), Sample->PlayerId);
            SerializeEntityList(*Sample, SObj);
            SamplesArray.Add(MakeShared<FJsonValueObject>(SObj));
        }

        if (SamplesArray.Num() == 0)
            continue;

        TSharedPtr<FJsonObject> PhaseObj = MakeShared<FJsonObject>();
        PhaseObj->SetNumberField(TEXT("PhaseId"), PhaseId);
        PhaseObj->SetArrayField(TEXT("samples"), SamplesArray);
        RootArray.Add(MakeShared<FJsonValueObject>(PhaseObj));
    }

    FString Output;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(RootArray, Writer);

    const FString Dir = FPaths::GetPath(ExportPath);
    if (!Dir.IsEmpty())
        IFileManager::Get().MakeDirectory(*Dir, true);

    return FFileHelper::SaveStringToFile(Output, *ExportPath);
}

// ================================================================
// GetSamplesInMemory
// ================================================================
int32 UAI_ReplayBufferManager::GetSamplesInMemory() const
{
    return Buffer.Num();
}

// ================================================================
// GetSamplesOnDisk
// ================================================================
int32 UAI_ReplayBufferManager::GetSamplesOnDisk() const
{
    int32 Count = 0;
    const FString ExportPath = GetTotalReplayBufferExportPath();

    FString FileData;
    if (!FFileHelper::LoadFileToString(FileData, *ExportPath))
        return 0;

    TSharedPtr<FJsonValue> RootValue;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);
    if (!FJsonSerializer::Deserialize(Reader, RootValue) ||
        !RootValue.IsValid() ||
        RootValue->Type != EJson::Array)
        return 0;

    for (const TSharedPtr<FJsonValue>& PhaseEntry : RootValue->AsArray())
    {
        const TSharedPtr<FJsonObject>* PhaseObjPtr = nullptr;
        if (!PhaseEntry.IsValid() || !PhaseEntry->TryGetObject(PhaseObjPtr))
            continue;
        const TArray<TSharedPtr<FJsonValue>>* SamplesArr = nullptr;
        if (!(*PhaseObjPtr)->TryGetArrayField(TEXT("samples"), SamplesArr))
            continue;
        Count += SamplesArr->Num();
    }

    return Count;
}

int32 UAI_ReplayBufferManager::GetSamplesInStagingFile() const
{
    const FString StagingPath = GetStagingPath();

    FString FileData;
    if (!FFileHelper::LoadFileToString(FileData, *StagingPath))
        return 0;

    TSharedPtr<FJsonValue> RootValue;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);
    if (!FJsonSerializer::Deserialize(Reader, RootValue) ||
        !RootValue.IsValid() ||
        RootValue->Type != EJson::Array)
        return 0;

    return RootValue->AsArray().Num();
}

// ================================================================
// FlushPartialToDisk
// ================================================================
bool UAI_ReplayBufferManager::FlushPartialToDisk()
{
    if (Buffer.Num() == 0)
        return true;

    const FString StagingPath = GetStagingPath();

    // ---- Load existing staging samples ----
    TArray<FMCTSTrainingSample> StagingBuffer;
    FString FileData;
    if (FFileHelper::LoadFileToString(FileData, *StagingPath))
    {
        TSharedPtr<FJsonValue> RootValue;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);
        if (FJsonSerializer::Deserialize(Reader, RootValue) &&
            RootValue.IsValid() &&
            RootValue->Type == EJson::Array)
        {
            for (const TSharedPtr<FJsonValue>& SampleVal : RootValue->AsArray())
            {
                const TSharedPtr<FJsonObject>* SObjPtr = nullptr;
                if (!SampleVal.IsValid() || !SampleVal->TryGetObject(SObjPtr))
                    continue;
                FMCTSTrainingSample Sample;
                DeserializeSample(*SObjPtr, Sample);
                StagingBuffer.Add(MoveTemp(Sample));
            }
        }
    }

    // ---- Append current buffer with placeholder value targets ----
    for (const FMCTSTrainingSample& S : Buffer)
    {
        FMCTSTrainingSample Copy = S;
        Copy.ValueTarget.Init(0.0f, NUM_PLAYERS);
        StagingBuffer.Add(MoveTemp(Copy));
    }

    // ---- Serialize flat array (no phase grouping in staging) ----
    TArray<TSharedPtr<FJsonValue>> RootArray;
    for (const FMCTSTrainingSample& Sample : StagingBuffer)
    {
        TSharedPtr<FJsonObject> SObj = MakeShared<FJsonObject>();
        SObj->SetArrayField(TEXT("node_features"),
            FloatArrayToJson(Sample.NodeFeatures));
        SObj->SetArrayField(TEXT("global_features"),
            FloatArrayToJson(Sample.GlobalFeatures));
        SObj->SetArrayField(TEXT("policy_target"),
            FloatArrayToJson(Sample.PolicyTarget));
        SObj->SetArrayField(TEXT("value_target"),
            FloatArrayToJson(Sample.ValueTarget));
        SObj->SetNumberField(TEXT("phase_id"), Sample.PhaseId);
        SObj->SetNumberField(TEXT("player_id"), Sample.PlayerId);
        TSharedPtr<FJsonObject> SampleObj = SObj;
        SerializeEntityList(Sample, SampleObj);
        RootArray.Add(MakeShared<FJsonValueObject>(SObj));
    }

    FString Output;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(RootArray, Writer);

    const FString Dir = FPaths::GetPath(StagingPath);
    if (!Dir.IsEmpty())
        IFileManager::Get().MakeDirectory(*Dir, true);

    const bool bSuccess = FFileHelper::SaveStringToFile(Output, *StagingPath);
    if (bSuccess)
        Buffer.Empty();

    return bSuccess;
}

// ================================================================
// ApplyOutcomeValuesToEpisode
// ================================================================
bool UAI_ReplayBufferManager::ApplyOutcomeValuesToEpisode(
    const TArray<float>& OutcomeValues)
{
    if (OutcomeValues.Num() != NUM_PLAYERS)
    {
        return false;
    }

    const FString StagingPath = GetStagingPath();

    // No staging file — all samples still in memory.
    // ExportReplayBuffer handles them normally.
    if (!FPaths::FileExists(StagingPath))
        return true;

    // ---- Load staging samples ----
    TArray<FMCTSTrainingSample> StagingBuffer;
    FString FileData;
    if (!FFileHelper::LoadFileToString(FileData, *StagingPath))
    {
        return false;
    }

    TSharedPtr<FJsonValue> RootValue;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);
    if (!FJsonSerializer::Deserialize(Reader, RootValue) ||
        !RootValue.IsValid() ||
        RootValue->Type != EJson::Array)
    {
        return false;
    }

    for (const TSharedPtr<FJsonValue>& SampleVal : RootValue->AsArray())
    {
        const TSharedPtr<FJsonObject>* SObjPtr = nullptr;
        if (!SampleVal.IsValid() || !SampleVal->TryGetObject(SObjPtr))
            continue;
        FMCTSTrainingSample Sample;
        DeserializeSample(*SObjPtr, Sample);
        StagingBuffer.Add(MoveTemp(Sample));
    }

    // ---- Apply actual outcome values ----
    for (FMCTSTrainingSample& Sample : StagingBuffer)
    {
        Sample.ValueTarget.SetNum(NUM_PLAYERS);
        for (int32 p = 0; p < NUM_PLAYERS; p++)
            Sample.ValueTarget[p] = OutcomeValues[p];
    }

    // ---- Drop invalid samples so the new-sample counter only counts
    //      samples that will actually be saved ----
    StagingBuffer.RemoveAll([this](const FMCTSTrainingSample& S)
        { return !IsValidTrainingSample(S); });

    // ---- Append corrected samples directly to TotalReplayBuffer ----
    TArray<FMCTSTrainingSample> TotalBuffer = LoadTotalBufferFromDisk();
    TotalBuffer.Append(StagingBuffer);
    NewSamplesSinceLastTraining += StagingBuffer.Num();
    SaveTrainingMetadata();
    EnforceCapacity(TotalBuffer);
    SerializeToDisk(TotalBuffer, GetTotalReplayBufferExportPath());

    // ---- Delete staging file ----
    IFileManager::Get().Delete(*StagingPath);

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

bool UAI_ReplayBufferManager::ExportReplayBuffer(
    const TArray<float>& FinalOutcomeValues)
{
    // ----------------------------------------------------------------
    // STEP 1: LOAD EXISTING TOTAL REPLAY BUFFER FROM DISK
    // ----------------------------------------------------------------
    TArray<FMCTSTrainingSample> TotalBuffer = LoadTotalBufferFromDisk();

    // ----------------------------------------------------------------
    // STEP 2: APPLY FINAL GAME REWARDS TO IN-MEMORY BUFFER ONLY
    // Staged samples were already written to disk with correct values
    // by ApplyOutcomeValuesToEpisode — do not apply again.
    // ----------------------------------------------------------------
    const int32 ExpectedPlayers = FinalOutcomeValues.Num();
    if (ExpectedPlayers <= 0)
    {
        ensureMsgf(false,
            TEXT("ExportReplayBuffer: invalid FinalOutcomeValues size"));
        return false;
    }

    for (FMCTSTrainingSample& Sample : Buffer)
    {
        Sample.ValueTarget.SetNumZeroed(ExpectedPlayers);
        for (int32 PlayerIdx = 0; PlayerIdx < ExpectedPlayers; ++PlayerIdx)
        {
            Sample.ValueTarget[PlayerIdx] =
                FinalOutcomeValues.IsValidIndex(PlayerIdx)
                ? FinalOutcomeValues[PlayerIdx]
                : 0.0f;
        }
    }

    // ----------------------------------------------------------------
    // STEP 3: MERGE IN-MEMORY BUFFER INTO TOTAL BUFFER
    // Invalid samples are dropped first so the new-sample counter
    // only counts samples that will actually be saved.
    // ----------------------------------------------------------------
    Buffer.RemoveAll([this](const FMCTSTrainingSample& S)
        { return !IsValidTrainingSample(S); });

    if (Buffer.Num() > 0)
    {
        TotalBuffer.Append(Buffer);
        NewSamplesSinceLastTraining += Buffer.Num();
        SaveTrainingMetadata();
    }
    Buffer.Empty();

    // ----------------------------------------------------------------
    // STEP 4: PRUNE
    // ----------------------------------------------------------------
    EnforceCapacity(TotalBuffer);

    // ----------------------------------------------------------------
    // STEP 5: VALIDATE
    // Iterate backwards so removals never skip or invalidate elements.
    // ----------------------------------------------------------------
    for (int32 i = TotalBuffer.Num() - 1; i >= 0; i--)
    {
        if (!IsValidTrainingSample(TotalBuffer[i]))
            TotalBuffer.RemoveAt(i);
    }

    // ----------------------------------------------------------------
    // STEP 6: SERIALIZE
    // ----------------------------------------------------------------
    return SerializeToDisk(TotalBuffer, GetTotalReplayBufferExportPath());
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
    // ------------------------------------------------------------
    Sample.ValueTarget.SetNum(NUM_PLAYERS);
    if (Root.bIsTerminal)
    {
        for (int32 p = 0; p < NUM_PLAYERS; ++p)
        {
            const int32 OutcomeIndex =
                6580 + GLOBAL_OUTCOME_VALUES + p;
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

int32 UAI_ReplayBufferManager::GetDiskSampleCount() const
{
    FString FileData;
    const FString ExportPath = GetTotalReplayBufferExportPath();

    if (!FFileHelper::LoadFileToString(FileData, *ExportPath))
    {
        return 0;
    }

    TSharedPtr<FJsonValue> RootValue;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);

    if (!FJsonSerializer::Deserialize(Reader, RootValue) ||
        !RootValue.IsValid() ||
        RootValue->Type != EJson::Array)
    {
        return 0;
    }

    int32 TotalCount = 0;

    for (const TSharedPtr<FJsonValue>& PhaseEntry : RootValue->AsArray())
    {
        const TSharedPtr<FJsonObject>* PhaseObjPtr = nullptr;
        if (!PhaseEntry.IsValid() || !PhaseEntry->TryGetObject(PhaseObjPtr))
            continue;

        const TArray<TSharedPtr<FJsonValue>>* SamplesArr = nullptr;
        if ((*PhaseObjPtr)->TryGetArrayField(TEXT("samples"), SamplesArr))
        {
            TotalCount += SamplesArr->Num();
        }
    }

    return TotalCount;
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