#include "UAI_RDGCache.h"
#include "NNE.h"
#include "NNERuntimeRDG.h"
#include "NNERuntimeCPU.h"
#include "AIInferenceSpec.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RHIGPUReadback.h"
#include "RHICommandList.h"
#include "RHIResources.h"
#include "RHIUtilities.h"
#include "RenderGraphValidation.h"

namespace
{
    FCriticalSection GModelSwapMutex;
}

UAI_RDGCache::FInferenceTimingStats UAI_RDGCache::TimingStats;

bool UAI_RDGCache::Initialize(const FString& RuntimeName)
{
    const FString ONNXPath = GetONNXPath();

    if (!FPaths::FileExists(ONNXPath))
    {
        ensureMsgf(false,
            TEXT("Inference Initialize failed: ONNX model not found at %s"), *ONNXPath);
        return false;
    }

    ModelData = LoadModelDataFromONNX(ONNXPath);
    if (!ModelData)
    {
        ensureMsgf(false,
            TEXT("Inference Initialize failed: could not load ModelData from %s"), *ONNXPath);
        return false;
    }

    bUseCPU = RuntimeName.EndsWith(TEXT("Cpu"));

    const bool bOk = bUseCPU
        ? RebuildCPUModelFromData(ModelData)
        : RebuildRDGModelFromData(ModelData);

    if (!bOk)
    {
        ensureMsgf(false, TEXT("Inference Initialize failed for runtime %s"), *RuntimeName);
        return false;
    }

    bModelInitialized = true;
    UE_LOG(LogTemp, Log, TEXT("Inference initialized with runtime %s"), *RuntimeName);
    return true;
}

void UAI_RDGCache::Reset()
{
    ReleaseCurrentModel();
    LastBatchSize = -1;
}

// ----------------------------------------------------------------
// RunInference — forwards to the CPU or GPU path chosen in Initialize.
// ----------------------------------------------------------------
bool UAI_RDGCache::RunInference(
    const TArray<TArray<float>>& NodeFeaturesBatch,
    const TArray<TArray<float>>& GlobalFeaturesBatch,
    const TArray<TArray<float>>& EntityTensorBatch,
    const TArray<TArray<float>>& EntityCountsBatch,
    const TArray<float>& PhaseIds,
    TArray<float>& OutHead2,
    TArray<float>& OutHead3,
    TArray<float>& OutHead7,
    TArray<float>& OutHead10,
    TArray<float>& OutHead13,
    TArray<float>& OutHead14,
    TArray<float>& OutHead20,
    TArray<float>& OutHead49,
    TArray<float>& OutHead128,
    TArray<float>& OutHead202,
    TArray<float>& OutHead330,
    TArray<float>& OutHead6581,
    TArray<float>& OutHead988,
    TArray<float>& OutValue)
{
    if (bUseCPU)
    {
        return RunInferenceCPU(
            NodeFeaturesBatch, GlobalFeaturesBatch, EntityTensorBatch, EntityCountsBatch, PhaseIds,
            OutHead2, OutHead3, OutHead7, OutHead10, OutHead13, OutHead14, OutHead20,
            OutHead49, OutHead128, OutHead202, OutHead330, OutHead6581, OutHead988, OutValue);
    }
    return RunInferenceGPU(
        NodeFeaturesBatch, GlobalFeaturesBatch, EntityTensorBatch, EntityCountsBatch, PhaseIds,
        OutHead2, OutHead3, OutHead7, OutHead10, OutHead13, OutHead14, OutHead20,
        OutHead49, OutHead128, OutHead202, OutHead330, OutHead6581, OutHead988, OutValue);
}

// ----------------------------------------------------------------
// RunInferenceCPU
// Runs the model synchronously on the calling thread (the search thread).
// No game-thread, render-thread or GPU synchronization.
// ----------------------------------------------------------------
bool UAI_RDGCache::RunInferenceCPU(
    const TArray<TArray<float>>& NodeFeaturesBatch,
    const TArray<TArray<float>>& GlobalFeaturesBatch,
    const TArray<TArray<float>>& EntityTensorBatch,
    const TArray<TArray<float>>& EntityCountsBatch,
    const TArray<float>& PhaseIds,
    TArray<float>& OutHead2, TArray<float>& OutHead3, TArray<float>& OutHead7,
    TArray<float>& OutHead10, TArray<float>& OutHead13, TArray<float>& OutHead14,
    TArray<float>& OutHead20, TArray<float>& OutHead49, TArray<float>& OutHead128,
    TArray<float>& OutHead202, TArray<float>& OutHead330, TArray<float>& OutHead6581,
    TArray<float>& OutHead988, TArray<float>& OutValue)
{
    FScopeLock Lock(&CPUInferenceMutex);

    if (!ModelInstanceCPU.IsValid())
        return false;

    const int32 BatchSize = PhaseIds.Num();
    const int32 NodeFlatSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;
    const int32 EntityFlatSize =
        NUM_TERRITORIES * MAX_UNIT_ENTITIES_PER_NODE * UNIT_ENTITY_FEATURE_COUNT;

    if (BatchSize <= 0 ||
        NodeFeaturesBatch.Num() != BatchSize ||
        GlobalFeaturesBatch.Num() != BatchSize ||
        EntityTensorBatch.Num() != BatchSize ||
        EntityCountsBatch.Num() != BatchSize)
    {
        ensureMsgf(false, TEXT("RunInferenceCPU: invalid batch input sizes"));
        return false;
    }

    // ---- Flatten inputs (every row must have its exact size) ----
    TArray<float> FlatNode, FlatGlobal, FlatEntity, FlatCounts;
    FlatNode.Reserve(BatchSize * NodeFlatSize);
    FlatGlobal.Reserve(BatchSize * GLOBAL_FEATURE_COUNT);
    FlatEntity.Reserve(BatchSize * EntityFlatSize);
    FlatCounts.Reserve(BatchSize * NUM_TERRITORIES);

    for (int32 i = 0; i < BatchSize; i++)
    {
        if (NodeFeaturesBatch[i].Num() != NodeFlatSize ||
            GlobalFeaturesBatch[i].Num() != GLOBAL_FEATURE_COUNT)
        {
            ensureMsgf(false, TEXT("RunInferenceCPU: bad feature size in batch row %d"), i);
            return false;
        }
        FlatNode.Append(NodeFeaturesBatch[i]);
        FlatGlobal.Append(GlobalFeaturesBatch[i]);

        // Empty entity data is allowed (treated as no units).
        if (EntityTensorBatch[i].Num() == EntityFlatSize)
            FlatEntity.Append(EntityTensorBatch[i]);
        else
            FlatEntity.AddZeroed(EntityFlatSize);

        if (EntityCountsBatch[i].Num() == NUM_TERRITORIES)
            FlatCounts.Append(EntityCountsBatch[i]);
        else
            FlatCounts.AddZeroed(NUM_TERRITORIES);
    }

    // ---- Input shapes (only when the batch size changes) ----
    if (CPUShapeBatchSize != BatchSize)
    {
        TArray<UE::NNE::FTensorShape> InputShapes;
        InputShapes.Add(UE::NNE::FTensorShape::Make({
            (uint32)BatchSize, (uint32)NUM_TERRITORIES, (uint32)NODE_FEATURE_COUNT }));
        InputShapes.Add(UE::NNE::FTensorShape::Make({
            (uint32)BatchSize, (uint32)GLOBAL_FEATURE_COUNT }));
        InputShapes.Add(UE::NNE::FTensorShape::Make({
            (uint32)BatchSize, (uint32)NUM_TERRITORIES,
            (uint32)MAX_UNIT_ENTITIES_PER_NODE, (uint32)UNIT_ENTITY_FEATURE_COUNT }));
        InputShapes.Add(UE::NNE::FTensorShape::Make({
            (uint32)BatchSize, (uint32)NUM_TERRITORIES, 1u }));

        if (ModelInstanceCPU->SetInputTensorShapes(InputShapes) != UE::NNE::EResultStatus::Ok)
        {
            UE_LOG(LogTemp, Error,
                TEXT("RunInferenceCPU: SetInputTensorShapes failed for BatchSize=%d"), BatchSize);
            CPUShapeBatchSize = -1;
            return false;
        }
        CPUShapeBatchSize = BatchSize;
    }

    // ---- Outputs, in model.py order: HEAD_NAMES + ["value"] ----
    OutHead2.SetNumUninitialized(BatchSize * PolicyHeadSize::Head2);
    OutHead3.SetNumUninitialized(BatchSize * PolicyHeadSize::Head3);
    OutHead7.SetNumUninitialized(BatchSize * PolicyHeadSize::Head7);
    OutHead10.SetNumUninitialized(BatchSize * PolicyHeadSize::Head10);
    OutHead13.SetNumUninitialized(BatchSize * PolicyHeadSize::Head13);
    OutHead14.SetNumUninitialized(BatchSize * PolicyHeadSize::Head14);
    OutHead20.SetNumUninitialized(BatchSize * PolicyHeadSize::Head20);
    OutHead49.SetNumUninitialized(BatchSize * PolicyHeadSize::Head49);
    OutHead128.SetNumUninitialized(BatchSize * PolicyHeadSize::Head128);
    OutHead202.SetNumUninitialized(BatchSize * PolicyHeadSize::Head202);
    OutHead330.SetNumUninitialized(BatchSize * PolicyHeadSize::Head330);
    OutHead6581.SetNumUninitialized(BatchSize * PolicyHeadSize::Head6581);
    OutHead988.SetNumUninitialized(BatchSize * PolicyHeadSize::Head988);
    OutValue.SetNumUninitialized(BatchSize * NUM_PLAYERS);

    auto Bind = [](TArray<float>& A) -> UE::NNE::FTensorBindingCPU
        {
            return UE::NNE::FTensorBindingCPU{ A.GetData(), (uint64)A.Num() * sizeof(float) };
        };

    TArray<UE::NNE::FTensorBindingCPU> Inputs;
    Inputs.Add(Bind(FlatNode));
    Inputs.Add(Bind(FlatGlobal));
    Inputs.Add(Bind(FlatEntity));
    Inputs.Add(Bind(FlatCounts));

    TArray<UE::NNE::FTensorBindingCPU> Outputs;
    Outputs.Add(Bind(OutHead2));
    Outputs.Add(Bind(OutHead3));
    Outputs.Add(Bind(OutHead7));
    Outputs.Add(Bind(OutHead10));
    Outputs.Add(Bind(OutHead13));
    Outputs.Add(Bind(OutHead14));
    Outputs.Add(Bind(OutHead20));
    Outputs.Add(Bind(OutHead49));
    Outputs.Add(Bind(OutHead128));
    Outputs.Add(Bind(OutHead202));
    Outputs.Add(Bind(OutHead330));
    Outputs.Add(Bind(OutHead6581));
    Outputs.Add(Bind(OutHead988));
    Outputs.Add(Bind(OutValue));

    if (ModelInstanceCPU->RunSync(Inputs, Outputs) != UE::NNE::EResultStatus::Ok)
    {
        UE_LOG(LogTemp, Error, TEXT("RunInferenceCPU: RunSync failed (BatchSize=%d)"), BatchSize);
        return false;
    }

    LastBatchSize = BatchSize;
    return true;
}

// ----------------------------------------------------------------
// RunInferenceGPU — GPU (DirectML) path through the render graph.
// ----------------------------------------------------------------
bool UAI_RDGCache::RunInferenceGPU(
    const TArray<TArray<float>>& NodeFeaturesBatch,
    const TArray<TArray<float>>& GlobalFeaturesBatch,
    const TArray<TArray<float>>& EntityTensorBatch,
    const TArray<TArray<float>>& EntityCountsBatch,
    const TArray<float>& PhaseIds,
    TArray<float>& OutHead2,
    TArray<float>& OutHead3,
    TArray<float>& OutHead7,
    TArray<float>& OutHead10,
    TArray<float>& OutHead13,
    TArray<float>& OutHead14,
    TArray<float>& OutHead20,
    TArray<float>& OutHead49,
    TArray<float>& OutHead128,
    TArray<float>& OutHead202,
    TArray<float>& OutHead330,
    TArray<float>& OutHead6581,
    TArray<float>& OutHead988,
    TArray<float>& OutValue)
{
    if (!ModelInstance.IsValid())
        return false;

    const double PrepStart = FPlatformTime::Seconds();

    const int32 BatchSize = PhaseIds.Num();
    if (BatchSize <= 0 ||
        NodeFeaturesBatch.Num() != BatchSize ||
        GlobalFeaturesBatch.Num() != BatchSize ||
        EntityTensorBatch.Num() != BatchSize ||
        EntityCountsBatch.Num() != BatchSize)
    {
        ensureMsgf(false, TEXT("RunInference: invalid batch input sizes"));
        return false;
    }

    const int32 S2 = BatchSize * PolicyHeadSize::Head2;
    const int32 S3 = BatchSize * PolicyHeadSize::Head3;
    const int32 S7 = BatchSize * PolicyHeadSize::Head7;
    const int32 S10 = BatchSize * PolicyHeadSize::Head10;
    const int32 S13 = BatchSize * PolicyHeadSize::Head13;
    const int32 S14 = BatchSize * PolicyHeadSize::Head14;
    const int32 S20 = BatchSize * PolicyHeadSize::Head20;
    const int32 S49 = BatchSize * PolicyHeadSize::Head49;
    const int32 S128 = BatchSize * PolicyHeadSize::Head128;
    const int32 S202 = BatchSize * PolicyHeadSize::Head202;
    const int32 S330 = BatchSize * PolicyHeadSize::Head330;
    const int32 S6581 = BatchSize * PolicyHeadSize::Head6581;
    const int32 S988 = BatchSize * PolicyHeadSize::Head988;
    const int32 SVal = BatchSize * NUM_PLAYERS;

    OutHead2.SetNumUninitialized(S2);
    OutHead3.SetNumUninitialized(S3);
    OutHead7.SetNumUninitialized(S7);
    OutHead10.SetNumUninitialized(S10);
    OutHead13.SetNumUninitialized(S13);
    OutHead14.SetNumUninitialized(S14);
    OutHead20.SetNumUninitialized(S20);
    OutHead49.SetNumUninitialized(S49);
    OutHead128.SetNumUninitialized(S128);
    OutHead202.SetNumUninitialized(S202);
    OutHead330.SetNumUninitialized(S330);
    OutHead6581.SetNumUninitialized(S6581);
    OutHead988.SetNumUninitialized(S988);
    OutValue.SetNumUninitialized(SVal);

    const int32 NodeFlatSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;
    const int32 EntityFlatSize =
        NUM_TERRITORIES * MAX_UNIT_ENTITIES_PER_NODE * UNIT_ENTITY_FEATURE_COUNT;
    const int32 CountFlatSize = NUM_TERRITORIES;

    TArray<float> FlatNodeFeatures;
    FlatNodeFeatures.Reserve(BatchSize * NodeFlatSize);
    for (const TArray<float>& Row : NodeFeaturesBatch)
        FlatNodeFeatures.Append(Row);

    TArray<float> FlatGlobalFeatures;
    FlatGlobalFeatures.Reserve(BatchSize * GLOBAL_FEATURE_COUNT);
    for (const TArray<float>& Row : GlobalFeaturesBatch)
        FlatGlobalFeatures.Append(Row);

    TArray<float> FlatEntityTensor;
    FlatEntityTensor.Reserve(BatchSize * EntityFlatSize);
    for (const TArray<float>& Row : EntityTensorBatch)
        FlatEntityTensor.Append(Row);

    TArray<float> FlatEntityCounts;
    FlatEntityCounts.Reserve(BatchSize * CountFlatSize);
    for (const TArray<float>& Row : EntityCountsBatch)
        FlatEntityCounts.Append(Row);

    // Stage timestamps, filled on the game and render threads.
    struct FStageTimes
    {
        double Posted = 0.0;        // work handed to the game thread
        double Started = 0.0;       // game thread began the work
        double AfterUpload = 0.0;   // first flush finished
        double AfterExecute = 0.0;  // second flush finished
    };
    FStageTimes Times;

    struct FNNEBuffers
    {
        TRefCountPtr<FRDGPooledBuffer> Node, Global, Entity, Counts;
        TRefCountPtr<FRDGPooledBuffer> H2, H3, H7, H10, H13, H14;
        TRefCountPtr<FRDGPooledBuffer> H20, H49, H128, H202;
        TRefCountPtr<FRDGPooledBuffer> H330, H6581, H988, Value;
        bool bShapesSet = false;
        bool bOutputsRead = false;   // true only when every output was read back
        double ReadbackSeconds = 0.0;
    };

    TSharedPtr<FNNEBuffers>                    Bufs = MakeShared<FNNEBuffers>();
    TSharedPtr<UE::NNE::IModelInstanceRDG>     CapturedInstance = ModelInstance;

    // ----------------------------------------------------------------
    // If we are NOT on the game thread, marshal render commands to the
    // game thread and wait for completion via a sync event.
    // ENQUEUE_RENDER_COMMAND and FlushRenderingCommands must only be
    // called from the game thread.
    // ----------------------------------------------------------------
    auto DoRenderWork = [&]()
        {
            Times.Started = FPlatformTime::Seconds();

            ENQUEUE_RENDER_COMMAND(AAI_AllocNNEBuffers)(
                [
                    CapturedInstance, Bufs,
                    FlatNodeFeatures, FlatGlobalFeatures,
                    FlatEntityTensor, FlatEntityCounts,
                    BatchSize,
                    S2, S3, S7, S10, S13, S14, S20,
                    S49, S128, S202, S330, S6581, S988, SVal
                ]
            (FRHICommandListImmediate& RHICmdList) mutable
                {
                    TArray<UE::NNE::FTensorShape> InputShapes;
                    InputShapes.Add(UE::NNE::FTensorShape::Make({
                        (uint32)BatchSize, (uint32)NUM_TERRITORIES,
                        (uint32)NODE_FEATURE_COUNT }));
                    InputShapes.Add(UE::NNE::FTensorShape::Make({
                        (uint32)BatchSize, (uint32)GLOBAL_FEATURE_COUNT }));
                    InputShapes.Add(UE::NNE::FTensorShape::Make({
                        (uint32)BatchSize,
                        (uint32)NUM_TERRITORIES,
                        (uint32)MAX_UNIT_ENTITIES_PER_NODE,
                        (uint32)UNIT_ENTITY_FEATURE_COUNT }));
                    InputShapes.Add(UE::NNE::FTensorShape::Make({
                        (uint32)BatchSize, (uint32)NUM_TERRITORIES, 1u }));

                    if (CapturedInstance->SetInputTensorShapes(InputShapes) !=
                        UE::NNE::EResultStatus::Ok)
                    {
                        UE_LOG(LogTemp, Error,
                            TEXT("SetInputTensorShapes failed for BatchSize=%d"),
                            BatchSize);
                        return;
                    }

                    Bufs->bShapesSet = true;

                    auto MakeDesc = [](int32 N) -> FRDGBufferDesc
                        {
                            return FRDGBufferDesc::CreateBufferDesc(sizeof(float), N);
                        };

                    FRDGBuilder Builder(RHICmdList);

                    FRDGBufferRef NodeBuf = Builder.CreateBuffer(
                        MakeDesc(FlatNodeFeatures.Num()), TEXT("NNE_Node"));
                    FRDGBufferRef GlobBuf = Builder.CreateBuffer(
                        MakeDesc(FlatGlobalFeatures.Num()), TEXT("NNE_Global"));
                    FRDGBufferRef EntBuf = Builder.CreateBuffer(
                        MakeDesc(FlatEntityTensor.Num()), TEXT("NNE_Entity"));
                    FRDGBufferRef CntBuf = Builder.CreateBuffer(
                        MakeDesc(FlatEntityCounts.Num()), TEXT("NNE_Counts"));

                    FRDGBufferRef B2 = Builder.CreateBuffer(MakeDesc(S2), TEXT("NNE_H2"));
                    FRDGBufferRef B3 = Builder.CreateBuffer(MakeDesc(S3), TEXT("NNE_H3"));
                    FRDGBufferRef B7 = Builder.CreateBuffer(MakeDesc(S7), TEXT("NNE_H7"));
                    FRDGBufferRef B10 = Builder.CreateBuffer(MakeDesc(S10), TEXT("NNE_H10"));
                    FRDGBufferRef B13 = Builder.CreateBuffer(MakeDesc(S13), TEXT("NNE_H13"));
                    FRDGBufferRef B14 = Builder.CreateBuffer(MakeDesc(S14), TEXT("NNE_H14"));
                    FRDGBufferRef B20 = Builder.CreateBuffer(MakeDesc(S20), TEXT("NNE_H20"));
                    FRDGBufferRef B49 = Builder.CreateBuffer(MakeDesc(S49), TEXT("NNE_H49"));
                    FRDGBufferRef B128 = Builder.CreateBuffer(MakeDesc(S128), TEXT("NNE_H128"));
                    FRDGBufferRef B202 = Builder.CreateBuffer(MakeDesc(S202), TEXT("NNE_H202"));
                    FRDGBufferRef B330 = Builder.CreateBuffer(MakeDesc(S330), TEXT("NNE_H330"));
                    FRDGBufferRef B6581 = Builder.CreateBuffer(MakeDesc(S6581), TEXT("NNE_H6581"));
                    FRDGBufferRef B988 = Builder.CreateBuffer(MakeDesc(S988), TEXT("NNE_H988"));
                    FRDGBufferRef BVal = Builder.CreateBuffer(MakeDesc(SVal), TEXT("NNE_Value"));

                    Builder.QueueBufferUpload(NodeBuf, FlatNodeFeatures.GetData(),
                        FlatNodeFeatures.Num() * sizeof(float), ERDGInitialDataFlags::None);
                    Builder.QueueBufferUpload(GlobBuf, FlatGlobalFeatures.GetData(),
                        FlatGlobalFeatures.Num() * sizeof(float), ERDGInitialDataFlags::None);
                    Builder.QueueBufferUpload(EntBuf, FlatEntityTensor.GetData(),
                        FlatEntityTensor.Num() * sizeof(float), ERDGInitialDataFlags::None);
                    Builder.QueueBufferUpload(CntBuf, FlatEntityCounts.GetData(),
                        FlatEntityCounts.Num() * sizeof(float), ERDGInitialDataFlags::None);

                    Bufs->Node = Builder.ConvertToExternalBuffer(NodeBuf);
                    Bufs->Global = Builder.ConvertToExternalBuffer(GlobBuf);
                    Bufs->Entity = Builder.ConvertToExternalBuffer(EntBuf);
                    Bufs->Counts = Builder.ConvertToExternalBuffer(CntBuf);
                    Bufs->H2 = Builder.ConvertToExternalBuffer(B2);
                    Bufs->H3 = Builder.ConvertToExternalBuffer(B3);
                    Bufs->H7 = Builder.ConvertToExternalBuffer(B7);
                    Bufs->H10 = Builder.ConvertToExternalBuffer(B10);
                    Bufs->H13 = Builder.ConvertToExternalBuffer(B13);
                    Bufs->H14 = Builder.ConvertToExternalBuffer(B14);
                    Bufs->H20 = Builder.ConvertToExternalBuffer(B20);
                    Bufs->H49 = Builder.ConvertToExternalBuffer(B49);
                    Bufs->H128 = Builder.ConvertToExternalBuffer(B128);
                    Bufs->H202 = Builder.ConvertToExternalBuffer(B202);
                    Bufs->H330 = Builder.ConvertToExternalBuffer(B330);
                    Bufs->H6581 = Builder.ConvertToExternalBuffer(B6581);
                    Bufs->H988 = Builder.ConvertToExternalBuffer(B988);
                    Bufs->Value = Builder.ConvertToExternalBuffer(BVal);

                    Builder.Execute();
                });

            FlushRenderingCommands();
            Times.AfterUpload = FPlatformTime::Seconds();

            TArray<float>* pH2 = &OutHead2;
            TArray<float>* pH3 = &OutHead3;
            TArray<float>* pH7 = &OutHead7;
            TArray<float>* pH10 = &OutHead10;
            TArray<float>* pH13 = &OutHead13;
            TArray<float>* pH14 = &OutHead14;
            TArray<float>* pH20 = &OutHead20;
            TArray<float>* pH49 = &OutHead49;
            TArray<float>* pH128 = &OutHead128;
            TArray<float>* pH202 = &OutHead202;
            TArray<float>* pH330 = &OutHead330;
            TArray<float>* pH6581 = &OutHead6581;
            TArray<float>* pH988 = &OutHead988;
            TArray<float>* pVal = &OutValue;

            ENQUEUE_RENDER_COMMAND(AAI_RunInference)(
                [
                    CapturedInstance, Bufs,
                    S2, S3, S7, S10, S13, S14, S20,
                    S49, S128, S202, S330, S6581, S988, SVal,
                    pH2, pH3, pH7, pH10, pH13, pH14, pH20,
                    pH49, pH128, pH202, pH330, pH6581, pH988, pVal
                ]
            (FRHICommandListImmediate& RHICmdList) mutable
                {
                    if (!Bufs->bShapesSet) return;

                    FRDGBuilder GraphBuilder(RHICmdList);

                    TArray<UE::NNE::FTensorBindingRDG> Inputs;
                    Inputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->Node) });
                    Inputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->Global) });
                    Inputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->Entity) });
                    Inputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->Counts) });

                    // Order must match HEAD_NAMES + ["value"] in model.py:
                    // head2, head3, head7, head10, head13, head14, head20,
                    // head49, head128, head202, head330, head6581, head988, value
                    TArray<UE::NNE::FTensorBindingRDG> Outputs;
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H2) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H3) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H7) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H10) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H13) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H14) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H20) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H49) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H128) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H202) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H330) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H6581) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->H988) });
                    Outputs.Add({ GraphBuilder.RegisterExternalBuffer(Bufs->Value) });

                    if (CapturedInstance->EnqueueRDG(GraphBuilder, Inputs, Outputs) !=
                        UE::NNE::EResultStatus::Ok)
                    {
                        UE_LOG(LogTemp, Error,
                            TEXT("Graph Transformer RDG enqueue failed"));
                        return;
                    }

                    GraphBuilder.Execute();
                    RHICmdList.BlockUntilGPUIdle();

                    auto ReadBack = [&](
                        TRefCountPtr<FRDGPooledBuffer>& Buf,
                        TArray<float>* Target, int32 Count)
                        {
                            void* Ptr = RHICmdList.LockBuffer(
                                Buf->GetRHI(), 0, Count * sizeof(float), RLM_ReadOnly);
                            FMemory::Memcpy(Target->GetData(), Ptr, Count * sizeof(float));
                            RHICmdList.UnlockBuffer(Buf->GetRHI());
                        };

                    const double ReadbackStart = FPlatformTime::Seconds();
                    ReadBack(Bufs->H2, pH2, S2);
                    ReadBack(Bufs->H3, pH3, S3);
                    ReadBack(Bufs->H7, pH7, S7);
                    ReadBack(Bufs->H10, pH10, S10);
                    ReadBack(Bufs->H13, pH13, S13);
                    ReadBack(Bufs->H14, pH14, S14);
                    ReadBack(Bufs->H20, pH20, S20);
                    ReadBack(Bufs->H49, pH49, S49);
                    ReadBack(Bufs->H128, pH128, S128);
                    ReadBack(Bufs->H202, pH202, S202);
                    ReadBack(Bufs->H330, pH330, S330);
                    ReadBack(Bufs->H6581, pH6581, S6581);
                    ReadBack(Bufs->H988, pH988, S988);
                    ReadBack(Bufs->Value, pVal, SVal);

                    Bufs->ReadbackSeconds = FPlatformTime::Seconds() - ReadbackStart;
                    Bufs->bOutputsRead = true;
                });

            FlushRenderingCommands();
            Times.AfterExecute = FPlatformTime::Seconds();
        };

    const double PrepSeconds = FPlatformTime::Seconds() - PrepStart;

    // ----------------------------------------------------------------
    // Execute render work on game thread if called from background thread
    // ----------------------------------------------------------------
    Times.Posted = FPlatformTime::Seconds();
    if (IsInGameThread())
    {
        DoRenderWork();
    }
    else
    {
        FEvent* SyncEvent = FPlatformProcess::GetSynchEventFromPool(false);

        AsyncTask(ENamedThreads::GameThread, [&DoRenderWork, SyncEvent]()
            {
                DoRenderWork();
                SyncEvent->Trigger();
            });

        SyncEvent->Wait();
        FPlatformProcess::ReturnSynchEventToPool(SyncEvent);
    }

    LastBatchSize = BatchSize;

    TimingStats.PrepSeconds += PrepSeconds;
    TimingStats.GameThreadWaitSeconds += Times.Started - Times.Posted;
    TimingStats.UploadSeconds += Times.AfterUpload - Times.Started;
    TimingStats.ExecuteSeconds +=
        (Times.AfterExecute - Times.AfterUpload) - Bufs->ReadbackSeconds;
    TimingStats.ReadbackSeconds += Bufs->ReadbackSeconds;

    // Outputs are only valid if shapes were set, the graph was enqueued,
    // and every buffer was read back.
    return Bufs->bOutputsRead;
}

bool UAI_RDGCache::RequestModelSwap(
    const FString& NewONNXPath,
    const FString& NewPTPath)
{
    if (!UAIModelArtifactManager::ValidateArtifactPair(NewPTPath, NewONNXPath))
    {
        ensureMsgf(false, TEXT("RDG swap blocked: artifact validation failed."));
        return false;
    }

    FModelArtifactSignature IncomingSignature =
        UAIModelArtifactManager::CreateSignature(NewPTPath, NewONNXPath);
    if (IncomingSignature.UnifiedVersionHash.IsEmpty())
    {
        ensureMsgf(false, TEXT("RDG swap blocked: invalid artifact signature."));
        return false;
    }

    // Called from the training thread; TickModelHotSwap reads these on the
    // game thread.
    FScopeLock Lock(&GModelSwapMutex);

    const bool bHasAcceptedArtifact =
        !LastAcceptedArtifactSignature.UnifiedVersionHash.IsEmpty();
    if (bHasAcceptedArtifact &&
        IncomingSignature.UnifiedVersionHash ==
        LastAcceptedArtifactSignature.UnifiedVersionHash)
    {
        ensureMsgf(false, TEXT("RDG swap blocked: artifact already active."));
        return false;
    }

    PendingArtifactSignature = IncomingSignature;
    PendingONNXPath = NewONNXPath;
    PendingPTPath = NewPTPath;
    bPendingModelSwap = true;
    return true;
}

bool UAI_RDGCache::ValidatePendingModelSwap()
{
    FScopeLock Lock(&GModelSwapMutex);

    if (!bPendingModelSwap)
        return false;

    if (PendingONNXPath.IsEmpty() || PendingPTPath.IsEmpty())
    {
        ensureMsgf(false,
            TEXT("Pending model swap contains empty artifact paths."));
        return false;
    }

    if (!UAIModelArtifactManager::ValidateArtifactPair(
        PendingPTPath, PendingONNXPath))
    {
        ensureMsgf(false,
            TEXT("Pending model swap failed artifact validation."));
        return false;
    }

    if (PendingArtifactSignature.UnifiedVersionHash.IsEmpty())
    {
        ensureMsgf(false,
            TEXT("Pending model swap contains invalid artifact signature."));
        return false;
    }

    if (!UAIModelArtifactManager::IsArtifactPairConsistent(
        PendingArtifactSignature))
    {
        ensureMsgf(false,
            TEXT("Pending model swap signature no longer matches artifacts on disk."));
        return false;
    }

    const FString CurrentUnifiedHash =
        UAIModelArtifactManager::ComputeUnifiedArtifactHash(
            PendingPTPath, PendingONNXPath);
    if (CurrentUnifiedHash.IsEmpty())
    {
        ensureMsgf(false, TEXT("Failed to recompute pending artifact hash."));
        return false;
    }

    if (CurrentUnifiedHash != PendingArtifactSignature.UnifiedVersionHash)
    {
        ensureMsgf(false,
            TEXT("Artifact changed after swap request was queued."));
        return false;
    }

    return true;
}

bool UAI_RDGCache::ApplyModelSwap(const FString& InPendingONNXPath)
{
    FScopeLock Lock(&ModelInstanceMutex);

    if (!ValidateModelArtifactSignature(InPendingONNXPath))
        return false;

    UNNEModelData* LoadedModelData = LoadModelDataFromONNX(InPendingONNXPath);
    if (!LoadedModelData)
        return false;

    ReleaseCurrentModel();

    const bool bRebuilt = bUseCPU
        ? RebuildCPUModelFromData(LoadedModelData)
        : RebuildRDGModelFromData(LoadedModelData);
    if (!bRebuilt)
        return false;

    PendingONNXPath = InPendingONNXPath;
    return true;
}

// ----------------------------------------------------------------
// TickModelHotSwap
// Call on the game thread while no inference is running (UAIManager
// calls it at the start of GetActionFromMCTS / GetActionFromAIModel).
// Loads the pending model and keeps it active.
// ----------------------------------------------------------------
void UAI_RDGCache::TickModelHotSwap()
{
    if (!bPendingModelSwap)
        return;

    auto ClearPending = [this]()
        {
            FScopeLock Lock(&GModelSwapMutex);
            PendingArtifactSignature = FModelArtifactSignature();
            PendingONNXPath.Reset();
            PendingPTPath.Reset();
            bPendingModelSwap = false;
        };

    if (!ValidatePendingModelSwap())
    {
        // Drop an invalid request instead of retrying it on every call.
        ClearPending();
        return;
    }

    FString SwapONNXPath;
    FModelArtifactSignature SwapSignature;
    {
        FScopeLock Lock(&GModelSwapMutex);
        SwapONNXPath = PendingONNXPath;
        SwapSignature = PendingArtifactSignature;
    }

    if (!ApplyModelSwap(SwapONNXPath))
    {
        ensureMsgf(false, TEXT("RDG swap failed during ApplyModelSwap."));
        ClearPending();
        return;
    }

    {
        FScopeLock Lock(&GModelSwapMutex);
        LastAcceptedArtifactSignature = SwapSignature;
    }
    ClearPending();

    // The new model instance stays active. (The previous version called
    // ResetInferenceState() here, which destroyed the model just loaded.)
    LastBatchSize = -1;

    UE_LOG(LogTemp, Log, TEXT("RDG model swap applied: %s"), *SwapONNXPath);
}

void UAI_RDGCache::FlushInferenceGraph()
{
    if (!ModelInstance.IsValid())
        return;

    FRHICommandListImmediate& RHICmdList =
        FRHICommandListExecutor::GetImmediateCommandList();
    RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);

    ENQUEUE_RENDER_COMMAND(FlushAIInferenceGraph)(
        [](FRHICommandListImmediate& CmdList)
        {
            CmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
        });

    FPlatformProcess::SleepNoStats(0.005f);
}

bool UAI_RDGCache::ReinitializeModelFromONNX(const FString& InONNXPath)
{
    if (InONNXPath.IsEmpty() || !FPaths::FileExists(InONNXPath))
        return false;

    UNNEModelData* NewModelData = LoadModelDataFromONNX(InONNXPath);
    if (!NewModelData)
        return false;

    ReleaseCurrentModel();
    return bUseCPU
        ? RebuildCPUModelFromData(NewModelData)
        : RebuildRDGModelFromData(NewModelData);
}

void UAI_RDGCache::ResetInferenceState()
{
    LastBatchSize = -1;
    ReleaseCurrentModel();
}

UNNEModelData* UAI_RDGCache::LoadModelDataFromONNX(const FString& InONNXPath)
{
    if (InONNXPath.IsEmpty())
        return nullptr;

    if (!FPaths::FileExists(InONNXPath))
    {
        UE_LOG(LogTemp, Error,
            TEXT("ONNX file not found: %s"), *InONNXPath);
        return nullptr;
    }

    TArray<uint8> LocalBytes;
    if (!FFileHelper::LoadFileToArray(LocalBytes, *InONNXPath))
    {
        UE_LOG(LogTemp, Error,
            TEXT("Failed to load ONNX file: %s"), *InONNXPath);
        return nullptr;
    }

    if (LocalBytes.Num() == 0)
    {
        UE_LOG(LogTemp, Error,
            TEXT("ONNX file empty: %s"), *InONNXPath);
        return nullptr;
    }

    ModelBytes = MoveTemp(LocalBytes);

    UNNEModelData* NewModelData = NewObject<UNNEModelData>(this);
    if (!NewModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to allocate UNNEModelData"));
        return nullptr;
    }

    const FString ModelType = TEXT("Onnx");
    const TConstArrayView64<uint8> ModelBufferView(ModelBytes);
    const TMap<FString, TConstArrayView64<uint8>> AdditionalBuffers;

    NewModelData->Init(ModelType, ModelBufferView, AdditionalBuffers);

    ModelData = NewModelData;
    return ModelData;
}

bool UAI_RDGCache::ValidateModelArtifactSignature(const FString& InONNXPath)
{
    if (InONNXPath.IsEmpty())
        return false;
    return FPaths::FileExists(InONNXPath);
}

bool UAI_RDGCache::RebuildRDGModelFromData(UNNEModelData* LoadedModelData)
{
    FScopeLock Lock(&ModelInstanceMutex);

    if (!LoadedModelData)
        return false;

    const FString RuntimeName = TEXT("NNERuntimeORTDml");

    TWeakInterfacePtr<INNERuntimeRDG> Runtime =
        UE::NNE::GetRuntime<INNERuntimeRDG>(RuntimeName);
    if (!Runtime.IsValid())
        return false;

    TSharedPtr<UE::NNE::IModelRDG> NewModelRDG =
        Runtime->CreateModelRDG(LoadedModelData);
    if (!NewModelRDG.IsValid())
        return false;

    ReleaseCurrentModel();
    ModelRDG = NewModelRDG;

    TSharedPtr<UE::NNE::IModelInstanceRDG> NewInstance =
        ModelRDG->CreateModelInstanceRDG();
    if (!NewInstance.IsValid())
    {
        ModelRDG.Reset();
        return false;
    }

    ModelInstance = NewInstance;
    LastBatchSize = -1;
    return true;
}

bool UAI_RDGCache::RebuildCPUModelFromData(UNNEModelData* LoadedModelData)
{
    FScopeLock Lock(&ModelInstanceMutex);

    if (!LoadedModelData)
        return false;

    TWeakInterfacePtr<INNERuntimeCPU> Runtime =
        UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
    if (!Runtime.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("NNERuntimeORTCpu runtime not available"));
        return false;
    }

    TSharedPtr<UE::NNE::IModelCPU> NewModelCPU = Runtime->CreateModelCPU(LoadedModelData);
    if (!NewModelCPU.IsValid())
        return false;

    TSharedPtr<UE::NNE::IModelInstanceCPU> NewInstance = NewModelCPU->CreateModelInstanceCPU();
    if (!NewInstance.IsValid())
        return false;

    FScopeLock CPULock(&CPUInferenceMutex);
    ReleaseCurrentModel();
    ModelCPU = NewModelCPU;
    ModelInstanceCPU = NewInstance;
    CPUShapeBatchSize = -1;
    LastBatchSize = -1;
    return true;
}

void UAI_RDGCache::ReleaseCurrentModel()
{
    FScopeLock Lock(&ModelInstanceMutex);
    ModelInstance.Reset();
    ModelRDG.Reset();

    FScopeLock CPULock(&CPUInferenceMutex);
    ModelInstanceCPU.Reset();
    ModelCPU.Reset();
    CPUShapeBatchSize = -1;
}