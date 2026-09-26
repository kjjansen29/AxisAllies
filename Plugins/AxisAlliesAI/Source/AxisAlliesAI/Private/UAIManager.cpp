// UAIManager.cpp

#include "UAIManager.h"
#include "UAI_RDGCache.h"
#include "UAI_InferenceQueue.h"
#include "AIInferenceSpec.h"

#include "Async/Async.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

#pragma once

static float GammaSample(FRandomStream& Rng, float Alpha)
{
    Alpha = FMath::Max(Alpha, 1e-5f);
    const float d = Alpha - 1.0f / 3.0f;
    const float c = 1.0f / FMath::Sqrt(9.0f * d);
    for (int32 MaxIter = 0; MaxIter < 1000; MaxIter++)
    {
        const float x = Rng.FRandRange(-1.0f, 1.0f);
        const float v = 1.0f + c * x;
        if (v <= 0.0f)
            continue;
        const float v3 = v * v * v;
        const float u = Rng.FRand();
        if (u < 1.0f - 0.0331f * (x * x) * (x * x))
            return d * v3;
        if (FMath::Loge(u) < 0.5f * x * x + d * (1.0f - v3 + FMath::Loge(v3)))
            return d * v3;
    }
    return d; // fallback if loop doesn't converge
}

static void SampleDirichlet(
    FRandomStream& Rng,
    const TArray<float>& AlphaVec,
    TArray<float>& OutNoise)
{
    const int32 N = AlphaVec.Num();
    OutNoise.SetNumUninitialized(N);

    float Sum = 0.0f;

    for (int32 i = 0; i < N; i++)
    {
        const float Alpha = FMath::Max(AlphaVec[i], 1e-5f);
        float Sample = GammaSample(Rng, Alpha);

        OutNoise[i] = Sample;
        Sum += Sample;
    }

    const float InvSum = (Sum > 0.0f) ? (1.0f / Sum) : 1.0f;

    for (int32 i = 0; i < N; i++)
    {
        OutNoise[i] *= InvSum;
    }
}

void UAIManager::ApplyRootDirichletNoise(FMCTSNode& Root, float Epsilon, float Alpha)
{
    const int32 ActionSize = Root.PolicyPrior.Num();
    if (ActionSize == 0)
        return;

    if (!IsTrainingMode())
    {
        return;
    }

    const float EffectiveEpsilon =
        (Epsilon > 0.0f)
        ? Epsilon
        : MCTSContext.DirichletEpsilon;

    const float EffectiveAlpha =
        (Alpha > 0.0f)
        ? Alpha
        : MCTSContext.DirichletAlpha;

    if (EffectiveEpsilon <= 0.0f || EffectiveAlpha <= 0.0f)
        return;

    FRandomStream Rng;

    // NEW: state-free deterministic root seeding using policy prior footprint
    uint32 Seed = 0;
    for (int32 i = 0; i < ActionSize; i++)
    {
        const float V = Root.PolicyPrior.IsValidIndex(i) ? Root.PolicyPrior[i] : 0.0f;
        Seed = FCrc::MemCrc32(&V, sizeof(float), Seed);
    }

    Rng.Initialize(Seed);

    TArray<float> AlphaVec;
    AlphaVec.SetNumUninitialized(ActionSize);

    for (int32 i = 0; i < ActionSize; i++)
    {
        AlphaVec[i] = EffectiveAlpha;
    }

    TArray<float> Noise;
    SampleDirichlet(Rng, AlphaVec, Noise);

    float NewSum = 0.0f;

    for (int32 i = 0; i < ActionSize; i++)
    {
        const bool bLegal =
            Root.LegalActionMask.IsValidIndex(i) && Root.LegalActionMask[i];

        const float Base =
            Root.PolicyPrior.IsValidIndex(i) ? Root.PolicyPrior[i] : 0.0f;

        const float MixedNoise = bLegal ? Noise[i] : 0.0f;

        Root.PolicyPrior[i] =
            (1.0f - EffectiveEpsilon) * Base +
            EffectiveEpsilon * MixedNoise;

        NewSum += Root.PolicyPrior[i];
    }

    const float InvNewSum = (NewSum > 0.0f) ? (1.0f / NewSum) : 1.0f;

    for (int32 i = 0; i < ActionSize; i++)
    {
        Root.PolicyPrior[i] *= InvNewSum;
    }
}

bool UAIManager::EnsureModelsExist()
{
    const FString BaseDir = GetPythonAIDir();
    const FString PtPath = GetPTPath();
    const FString OnnxPath = GetONNXPath();

    if (IFileManager::Get().FileExists(*PtPath) &&
        IFileManager::Get().FileExists(*OnnxPath))
    {
        return true;
    }

    UE_LOG(LogTemp, Warning,
        TEXT("EnsureModelsExist: model files not found — attempting to generate via model.py"));
    UE_LOG(LogTemp, Warning, TEXT("  PT path:   %s"), *PtPath);
    UE_LOG(LogTemp, Warning, TEXT("  ONNX path: %s"), *OnnxPath);
    UE_LOG(LogTemp, Warning, TEXT("  BaseDir:   %s"), *BaseDir);

    const FString PythonExe = TEXT("python");
    const FString Script = FPaths::Combine(BaseDir, TEXT("model.py"));
    const FString Args = FString::Printf(TEXT("\"%s\""), *Script);

    if (!FPaths::FileExists(Script))
    {
        UE_LOG(LogTemp, Error,
            TEXT("EnsureModelsExist: model.py not found at %s"), *Script);
        return false;
    }

    void* ReadPipe = nullptr;
    void* WritePipe = nullptr;
    FPlatformProcess::CreatePipe(ReadPipe, WritePipe);

    FProcHandle ProcHandle = FPlatformProcess::CreateProc(
        *PythonExe,
        *Args,
        true,
        false,
        false,
        nullptr,
        0,
        *BaseDir,
        WritePipe,
        ReadPipe
    );

    if (!ProcHandle.IsValid())
    {
        UE_LOG(LogTemp, Error,
            TEXT("EnsureModelsExist: failed to launch python process"));
        FPlatformProcess::ClosePipe(ReadPipe, WritePipe);
        return false;
    }

    FPlatformProcess::WaitForProc(ProcHandle);
    FPlatformProcess::CloseProc(ProcHandle);
    FPlatformProcess::ClosePipe(ReadPipe, WritePipe);

    const bool bPTExists = IFileManager::Get().FileExists(*PtPath);
    const bool bONNXExists = IFileManager::Get().FileExists(*OnnxPath);

    if (!bPTExists || !bONNXExists)
    {
        UE_LOG(LogTemp, Error,
            TEXT("EnsureModelsExist: model files still missing after running model.py"));
        UE_LOG(LogTemp, Error,
            TEXT("  PT exists:   %s"), bPTExists ? TEXT("YES") : TEXT("NO"));
        UE_LOG(LogTemp, Error,
            TEXT("  ONNX exists: %s"), bONNXExists ? TEXT("YES") : TEXT("NO"));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("EnsureModelsExist: model files generated successfully"));
    return true;
}

void UAIManager::BeginMCTS(
    const TArray<float>& GameState,
    int32 PhaseId,
    int32 PlayerId,
    UObject* InGameState,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    if (!EnsureModelsExist())
    {
        ResetMCTS();
        return;
    }

    MCTSContext.bRootInitialized = false;
    GameStateRef = InGameState;
    MCTSRandStream.Initialize(FPlatformTime::Cycles());

    if (!InferenceQueue)
        InferenceQueue = NewObject<UAI_InferenceQueue>(this);
    else
        InferenceQueue->Clear();

    // Requests from a previous search were answered or discarded with the
    // queue; nodes must not stay marked as waiting for an answer.
    InFlightNodes.Reset();

    if (!RDGCache)
    {
        RDGCache = NewObject<UAI_RDGCache>(this);
        RDGCache->Initialize(UAI_RDGCache::InferenceRuntimeName);
    }

    const int32 ExistingRoot =
        (Tree.Nodes.IsValidIndex(Tree.RootIndex) &&
            Tree.GetNode(Tree.RootIndex).PhaseId == PhaseId &&
            Tree.GetNode(Tree.RootIndex).PlayerId == PlayerId)
        ? Tree.RootIndex
        : INDEX_NONE;

    // ----------------------------------------------------------------
    // Per-simulation state for the pending-set batched MCTS
    // ----------------------------------------------------------------
    struct FSimState
    {
        TArray<int32> Path;
        int32         Current;
        bool          bComplete;
        bool          bPending;

        FSimState()
            : Current(INDEX_NONE)
            , bComplete(false)
            , bPending(false)
        {
        }
    };

    // ----------------------------------------------------------------
    // RunSimulations — pending-set batched MCTS with virtual loss.
    // ----------------------------------------------------------------
    auto RunSimulations = [&](int32 RootIdx, int32 NumSims)
        {

            TArray<FSimState> Sims;
            Sims.SetNum(NumSims);
            for (FSimState& S : Sims)
            {
                S.Current = RootIdx;
                S.bComplete = false;
                S.bPending = false;
            }

            int32 NumComplete = 0;

            while (NumComplete < NumSims)
            {
                // ---- CANCELLATION CHECK ----
                if (bCancelRequested)
                {
                    for (FSimState& S : Sims)
                    {
                        if (S.bComplete)
                            continue;
                        for (int32 p = 0; p + 1 < S.Path.Num(); p++)
                        {
                            if (!Tree.Nodes.IsValidIndex(S.Path[p]))
                                continue;
                            FMCTSNode& N = Tree.GetNode(S.Path[p]);
                            N.VirtualLossCount =
                                FMath::Max(0, N.VirtualLossCount - 1);
                            if (Tree.Nodes.IsValidIndex(S.Path[p + 1]))
                            {
                                const int32 A =
                                    Tree.GetNode(S.Path[p + 1]).ActionFromParent;
                                if (N.VirtualLossEdgeCount.IsValidIndex(A))
                                    N.VirtualLossEdgeCount[A] =
                                    FMath::Max(0,
                                        N.VirtualLossEdgeCount[A] - 1);
                            }
                        }
                        if (S.Path.Num() > 0 &&
                            Tree.Nodes.IsValidIndex(S.Path.Last()))
                            Backpropagate(S.Path.Last(), PlayerId);
                        S.bComplete = true;
                        NumComplete++;
                    }
                    break;
                }
                // ---- END CANCELLATION CHECK ----

                bool bAnyExpanded = false;

                for (FSimState& S : Sims)
                {
                    if (S.bComplete)
                        continue;

                    if (S.bPending)
                    {
                        const int32 BlockedNode =
                            S.Path.Num() > 0 ? S.Path.Last() : INDEX_NONE;

                        if (!Tree.Nodes.IsValidIndex(BlockedNode))
                        {
                            S.bComplete = true;
                            NumComplete++;
                            continue;
                        }

                        const FMCTSNode& BN = Tree.GetNode(BlockedNode);
                        if (!BN.bPolicyInitialized && !BN.bIsTerminal)
                            continue;

                        // Leaf has been evaluated: back up its value and
                        // finish this simulation (do not keep descending).
                        for (int32 p = 0; p + 1 < S.Path.Num(); p++)
                        {
                            if (!Tree.Nodes.IsValidIndex(S.Path[p]))
                                continue;
                            FMCTSNode& N = Tree.GetNode(S.Path[p]);
                            N.VirtualLossCount =
                                FMath::Max(0, N.VirtualLossCount - 1);
                            if (Tree.Nodes.IsValidIndex(S.Path[p + 1]))
                            {
                                const int32 A =
                                    Tree.GetNode(S.Path[p + 1]).ActionFromParent;
                                if (N.VirtualLossEdgeCount.IsValidIndex(A))
                                    N.VirtualLossEdgeCount[A] =
                                    FMath::Max(0,
                                        N.VirtualLossEdgeCount[A] - 1);
                            }
                        }
                        Backpropagate(BlockedNode, PlayerId);
                        S.bPending = false;
                        S.bComplete = true;
                        NumComplete++;
                        continue;
                    }

                    // ---- Traverse down the tree ----
                    while (true)
                    {
                        if (!Tree.Nodes.IsValidIndex(S.Current))
                        {
                            for (int32 p = 0; p + 1 < S.Path.Num(); p++)
                            {
                                if (!Tree.Nodes.IsValidIndex(S.Path[p]))
                                    continue;
                                FMCTSNode& N = Tree.GetNode(S.Path[p]);
                                N.VirtualLossCount =
                                    FMath::Max(0, N.VirtualLossCount - 1);
                                if (Tree.Nodes.IsValidIndex(S.Path[p + 1]))
                                {
                                    const int32 A =
                                        Tree.GetNode(S.Path[p + 1]).ActionFromParent;
                                    if (N.VirtualLossEdgeCount.IsValidIndex(A))
                                        N.VirtualLossEdgeCount[A] =
                                        FMath::Max(0,
                                            N.VirtualLossEdgeCount[A] - 1);
                                }
                            }
                            if (S.Path.Num() > 0 &&
                                Tree.Nodes.IsValidIndex(S.Path.Last()))
                                Backpropagate(S.Path.Last(), PlayerId);
                            S.bComplete = true;
                            NumComplete++;
                            break;
                        }

                        FMCTSNode& Node = Tree.GetNode(S.Current);
                        ValidateMCTSRuntimeInvariants(Node);

                        if (Node.bIsTerminal)
                        {
                            S.Path.Add(S.Current);
                            for (int32 p = 0; p + 1 < S.Path.Num(); p++)
                            {
                                if (!Tree.Nodes.IsValidIndex(S.Path[p]))
                                    continue;
                                FMCTSNode& N = Tree.GetNode(S.Path[p]);
                                N.VirtualLossCount =
                                    FMath::Max(0, N.VirtualLossCount - 1);
                                if (Tree.Nodes.IsValidIndex(S.Path[p + 1]))
                                {
                                    const int32 A =
                                        Tree.GetNode(S.Path[p + 1]).ActionFromParent;
                                    if (N.VirtualLossEdgeCount.IsValidIndex(A))
                                        N.VirtualLossEdgeCount[A] =
                                        FMath::Max(0,
                                            N.VirtualLossEdgeCount[A] - 1);
                                }
                            }
                            Backpropagate(S.Current, PlayerId);
                            S.bComplete = true;
                            NumComplete++;
                            break;
                        }

                        if (!Node.bPolicyInitialized)
                        {
                            if (S.Path.Num() == 0 ||
                                S.Path.Last() != S.Current)
                                S.Path.Add(S.Current);
                            S.bPending = true;
                            break;
                        }

                        if (S.Path.Num() == 0 || S.Path.Last() != S.Current)
                            S.Path.Add(S.Current);

                        const int32 Action = SelectActionPUCT(S.Current, 1.5f);
                        if (Action < 0)
                        {
                            // Add current node to path before backpropagating
                            if (S.Path.Num() == 0 || S.Path.Last() != S.Current)
                                S.Path.Add(S.Current);
                            for (int32 p = 0; p + 1 < S.Path.Num(); p++)
                            {
                                if (!Tree.Nodes.IsValidIndex(S.Path[p]))
                                    continue;
                                FMCTSNode& N = Tree.GetNode(S.Path[p]);
                                N.VirtualLossCount =
                                    FMath::Max(0, N.VirtualLossCount - 1);
                                if (Tree.Nodes.IsValidIndex(S.Path[p + 1]))
                                {
                                    const int32 A =
                                        Tree.GetNode(S.Path[p + 1]).ActionFromParent;
                                    if (N.VirtualLossEdgeCount.IsValidIndex(A))
                                        N.VirtualLossEdgeCount[A] =
                                        FMath::Max(0,
                                            N.VirtualLossEdgeCount[A] - 1);
                                }
                            }
                            if (S.Path.Num() > 0 &&
                                Tree.Nodes.IsValidIndex(S.Path.Last()))
                                Backpropagate(S.Path.Last(), PlayerId);
                            S.bComplete = true;
                            NumComplete++;
                            break;
                        }

                        if (!Node.VirtualLossEdgeCount.IsValidIndex(Action))
                            Node.VirtualLossEdgeCount.SetNum(
                                FMath::Max(
                                    Node.VirtualLossEdgeCount.Num(),
                                    Action + 1));
                        Node.VirtualLossEdgeCount[Action]++;
                        Node.VirtualLossCount++;

                        int32 Next = INDEX_NONE;
                        for (int32 ChildIdx : Node.ChildIndices)
                        {
                            if (Tree.GetNode(ChildIdx).ActionFromParent == Action)
                            {
                                Next = ChildIdx;
                                break;
                            }
                        }

                        if (Next == INDEX_NONE)
                        {
                            ExpandNode(S.Current, Action);

                            const FMCTSNode& ExpandedParent =
                                Tree.GetNode(S.Current);
                            const int32 ChildIndex =
                                ExpandedParent.ChildIndices.Num() > 0
                                ? ExpandedParent.ChildIndices.Last()
                                : INDEX_NONE;

                            if (ChildIndex != INDEX_NONE)
                            {
                                S.Path.Add(ChildIndex);
                                S.Current = ChildIndex;
                                S.bPending = true;
                            }
                            else
                            {
                                for (int32 p = 0; p + 1 < S.Path.Num(); p++)
                                {
                                    if (!Tree.Nodes.IsValidIndex(S.Path[p]))
                                        continue;
                                    FMCTSNode& N = Tree.GetNode(S.Path[p]);
                                    N.VirtualLossCount =
                                        FMath::Max(0, N.VirtualLossCount - 1);
                                    if (Tree.Nodes.IsValidIndex(S.Path[p + 1]))
                                    {
                                        const int32 A =
                                            Tree.GetNode(
                                                S.Path[p + 1]).ActionFromParent;
                                        if (N.VirtualLossEdgeCount.IsValidIndex(A))
                                            N.VirtualLossEdgeCount[A] =
                                            FMath::Max(0,
                                                N.VirtualLossEdgeCount[A] - 1);
                                    }
                                }
                                S.bComplete = true;
                                NumComplete++;
                            }

                            bAnyExpanded = true;
                            break;
                        }

                        S.Current = Next;
                    }
                }

                // ---- Flush when all remaining sims are pending ----
                const int32 QueueSize =
                    InferenceQueue ? InferenceQueue->GetQueueSize() : 0;

                bool bAllRemainingPending = true;
                for (const FSimState& S : Sims)
                {
                    if (!S.bComplete && !S.bPending)
                    {
                        bAllRemainingPending = false;
                        break;
                    }
                }

                // Flush when every remaining simulation is waiting for
                // inference, or when enough requests have queued up.
                constexpr int32 InferenceFlushThreshold = 64;
                if (QueueSize > 0 && (bAllRemainingPending || QueueSize >= InferenceFlushThreshold))
                    FlushInferenceBatch();

                if (!bAnyExpanded && bAllRemainingPending && QueueSize == 0)
                {
                    bool bAnyStillPending = false;
                    for (const FSimState& S : Sims)
                    {
                        if (!S.bComplete && S.bPending)
                        {
                            bAnyStillPending = true;
                            break;
                        }
                    }
                    if (bAnyStillPending)
                    {
                        for (FSimState& S : Sims)
                        {
                            if (S.bComplete)
                                continue;
                            for (int32 p = 0; p + 1 < S.Path.Num(); p++)
                            {
                                if (!Tree.Nodes.IsValidIndex(S.Path[p]))
                                    continue;
                                FMCTSNode& N = Tree.GetNode(S.Path[p]);
                                N.VirtualLossCount =
                                    FMath::Max(0, N.VirtualLossCount - 1);
                                if (Tree.Nodes.IsValidIndex(S.Path[p + 1]))
                                {
                                    const int32 A =
                                        Tree.GetNode(S.Path[p + 1]).ActionFromParent;
                                    if (N.VirtualLossEdgeCount.IsValidIndex(A))
                                        N.VirtualLossEdgeCount[A] =
                                        FMath::Max(0,
                                            N.VirtualLossEdgeCount[A] - 1);
                                }
                            }
                            if (S.Path.Num() > 0 &&
                                Tree.Nodes.IsValidIndex(S.Path.Last()))
                                Backpropagate(S.Path.Last(), PlayerId);
                            S.bComplete = true;
                            NumComplete++;
                        }
                    }
                }
            }
        };


    // ----------------------------------------------------------------
    // Run simulations in waves. Simulations within a wave run together
    // (their leaf evaluations are batched); each wave starts from the
    // tree as updated by the previous waves, so later simulations use
    // earlier results. Larger waves = bigger inference batches but less
    // sequential refinement.
    // ----------------------------------------------------------------
    // One wave fills one inference batch.
    constexpr int32 SimulationsPerWave = INFERENCE_BATCH_SIZE;
    auto RunSimulationWaves = [&](int32 RootIdx, int32 TotalSims)
        {
            if (!Tree.Nodes.IsValidIndex(RootIdx))
                return;

            // ---- Forced move: only one legal action ----
            // One simulation is enough: it creates the child node, so the
            // tree (and its pending context) is reused for the next decision.
            {
                const FMCTSNode& R = Tree.GetNode(RootIdx);
                int32 LegalCount = 0;
                for (bool bLegal : R.LegalActionMask)
                    if (bLegal) LegalCount++;
                if (LegalCount <= 1)
                    TotalSims = FMath::Min(TotalSims, 1);
            }

            for (int32 Done = 0; Done < TotalSims && !bCancelRequested; )
            {
                const int32 WaveSims = FMath::Min(SimulationsPerWave, TotalSims - Done);
                RunSimulations(RootIdx, WaveSims);
                Done += WaveSims;

                // ---- Early stop: the result can no longer change ----
                // If the most-visited root action leads the runner-up by more
                // than the simulations left, the remaining ones cannot change
                // which action is best.
                const int32 Remaining = TotalSims - Done;
                if (Remaining <= 0 || !Tree.Nodes.IsValidIndex(RootIdx))
                    continue;

                const FMCTSNode& R = Tree.GetNode(RootIdx);
                int32 Best = 0;
                int32 Second = 0;
                for (int32 a = 0; a < R.EdgeVisitCount.Num(); a++)
                {
                    if (!R.LegalActionMask.IsValidIndex(a) || !R.LegalActionMask[a])
                        continue;
                    const int32 V = R.EdgeVisitCount[a];
                    if (V > Best) { Second = Best; Best = V; }
                    else if (V > Second) { Second = V; }
                }
                if (Best - Second > Remaining)
                    break;
            }
        };

    // ----------------------------------------------------------------
    // EXISTING ROOT PATH
    // ----------------------------------------------------------------
    if (ExistingRoot != INDEX_NONE && Tree.Nodes.IsValidIndex(ExistingRoot))
    {
        Tree.RootIndex = ExistingRoot;

        if (PendingRootActionContext.bIsValid)
        {
            Tree.Nodes[ExistingRoot].PendingActionContext = PendingRootActionContext;
            for (int32 ChildIdx : Tree.Nodes[ExistingRoot].ChildIndices)
            {
                if (Tree.Nodes.IsValidIndex(ChildIdx))
                    Tree.Nodes[ChildIdx].PendingActionContext.PendingActionA =
                    PendingRootActionContext.PendingActionA;
            }
            PendingRootActionContext = FPendingActionContext();
        }

        const int32 RootIndexForBudget = Tree.RootIndex;
        const int32 NumSimulations =
            GetAdaptiveSimulationCount(RootIndexForBudget);
        Tree.Nodes.Reserve(Tree.Nodes.Num() + NumSimulations * 10);
        FMCTSNode& RootNode = Tree.GetNode(ExistingRoot);

        EvaluateNode(ExistingRoot);
        FlushInferenceBatch();

        if (!RootNode.bPolicyInitialized)
            return;

        if (RootNode.LegalActionMask.Num() == 0)
            return;

        RootNode.VisitCount = FMath::Max(RootNode.VisitCount, 1);

        if (IsTrainingMode())
            ApplyRootDirichletNoise(RootNode, 0.25f, 0.3f);

        //UE_LOG(LogTemp, Warning,
            //TEXT("BeginMCTS: RunSimulations starting (existing root). TreeNodes=%d NumSims=%d"),
            //Tree.Nodes.Num(), NumSimulations);
        RunSimulationWaves(RootIndexForBudget, NumSimulations);
        FlushInferenceBatch();
        //UE_LOG(LogTemp, Warning,
            //TEXT("ExistingRoot: bPolicyInitialized=%s LegalMaskNum=%d VisitCount=%d"),
            //RootNode.bPolicyInitialized ? TEXT("true") : TEXT("false"),
            //RootNode.LegalActionMask.Num(),
            //RootNode.VisitCount);
        //UE_LOG(LogTemp, Warning,
            //TEXT("BeginMCTS: RunSimulations complete (existing root). TreeNodes=%d"),
            //Tree.Nodes.Num());
        return;
    }

    // ----------------------------------------------------------------
    // NEW ROOT PATH
    // ----------------------------------------------------------------
    ResetMCTS();
    Tree = FMCTSTree();

    const int32 RootIndex =
        CreateRootNode(GameState, PhaseId, PlayerId, InEntityLists);

    if (!Tree.Nodes.IsValidIndex(RootIndex))
        return;

    Tree.RootIndex = RootIndex;

    const int32 NumSimulations = GetAdaptiveSimulationCount(RootIndex);
    Tree.Nodes.Reserve(NumSimulations * 10);

    FMCTSNode& RootNode = Tree.GetNode(RootIndex);

    ensureMsgf(
        RootNode.NodeFeatures.Num() ==
        NUM_TERRITORIES * NODE_FEATURE_COUNT &&
        RootNode.GlobalFeatures.Num() == GLOBAL_FEATURE_COUNT,
        TEXT("Root feature vectors invalid size")
    );
    ensureMsgf(
        RootNode.LegalActionMask.Num() > 0,
        TEXT("Root LegalActionMask missing")
    );

    EvaluateNode(RootIndex);
    FlushInferenceBatch();

    if (!RootNode.bPolicyInitialized)
        return;

    RootNode.VisitCount = 1;

    if (IsTrainingMode())
        ApplyRootDirichletNoise(RootNode, 0.25f, 0.3f);

    RunSimulationWaves(RootIndex, NumSimulations);
    FlushInferenceBatch();
}


int32 UAIManager::CreateRootNode(
    const TArray<float>& GameState,
    int32 PhaseId,
    int32 PlayerId,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    FMCTSNode RootNode;
    RootNode.NodeFeatures = ExtractNodeFeatures(GameState);
    RootNode.GlobalFeatures = ExtractGlobalFeatures(GameState);
    RootNode.PhaseId = PhaseId;
    RootNode.PlayerId = PlayerId;
    RootNode.EntityLists = InEntityLists;

    // Ensure EntityLists has the correct number of territories
    if (RootNode.EntityLists.Num() != NUM_TERRITORIES)
    {
        RootNode.EntityLists.SetNum(NUM_TERRITORIES);
    }

    const int32 ActionSpaceSize =
        GetPolicySizeForPhase(static_cast<EPhaseId>(PhaseId));

    RootNode.PolicyPrior.Init(0.0f, ActionSpaceSize);
    RootNode.EdgeVisitCount.Init(0, ActionSpaceSize);
    RootNode.VirtualLossEdgeCount.Init(0, ActionSpaceSize);
    RootNode.EdgeValueSum.Init(0.0f, ActionSpaceSize);
    RootNode.VisitCount = 0;
    RootNode.TotalValue = 0.0f;
    RootNode.bIsExpanded = false;
    RootNode.bPolicyInitialized = false;
    RootNode.bIsTerminal = false;
    RootNode.MCTSValuePerPlayer.Init(0.0f, NumPlayers);

    // Apply pending action context if set via SetPendingActionContext
    if (PendingRootActionContext.bIsValid)
    {
        RootNode.PendingActionContext = PendingRootActionContext;
    }

    const int32 RootIndex = Tree.CreateNode(RootNode);
    Tree.RootIndex = RootIndex;

    const bool bHasPending = PendingRootActionContext.bIsValid;
    const FApplyActionResult MaskResult = Internal_GetLegalActionMask(
        GameState,
        PhaseId,
        PlayerId,
        bHasPending,
        bHasPending ? PendingRootActionContext.PendingActionA : INDEX_NONE,
        bHasPending ? PendingRootActionContext.PendingActionB : INDEX_NONE,
        bHasPending ? PendingRootActionContext.PendingActionC : INDEX_NONE,
        bHasPending ? PendingRootActionContext.PendingActionD : INDEX_NONE,
        RootIndex,
        Tree.Nodes[RootIndex].EntityLists
    );

    Tree.Nodes[RootIndex].LegalActionMask = MaskResult.OutLegalActionMask;
    Tree.Nodes[RootIndex].bIsTerminal = MaskResult.bIsTerminal;

    // Reset pending action context after use
    PendingRootActionContext = FPendingActionContext();

    ensureMsgf(
        Tree.Nodes[RootIndex].LegalActionMask.Num() == ActionSpaceSize,
        TEXT("CreateRootNode: GetLegalActionMask returned mask of size %d, expected %d for PhaseId %d"),
        Tree.Nodes[RootIndex].LegalActionMask.Num(), ActionSpaceSize, PhaseId
    );

    return RootIndex;
}

void UAIManager::ExpandNode(int32 NodeIndex, int32 Action)
{
    // Copy parent node by value immediately to avoid dangling references
    // if Tree.Nodes reallocates during a subsequent CreateNode call.
    const FMCTSNode Parent = Tree.GetNode(NodeIndex);
    Tree.GetNode(NodeIndex).bIsExpanded = true;

    TArray<float> CombinedState;
    CombinedState.Reserve(Parent.NodeFeatures.Num() + Parent.GlobalFeatures.Num());
    CombinedState.Append(Parent.NodeFeatures);
    CombinedState.Append(Parent.GlobalFeatures);

    const EPhaseId ParentPhase = static_cast<EPhaseId>(Parent.PhaseId);
    const int32    ParentPlayerId = Parent.PlayerId;

    FMCTSNode Child;
    int32 ChildIndex = INDEX_NONE;

    if (IsChainIntermediatePhase(ParentPhase))
    {
        // Chain-intermediate phases that change the game state must still
        // run SimulateTransition, so the next chain step sees the updated
        // state and entity lists. All other chain-intermediate phases copy
        // state and entity lists forward unchanged.
        const bool bSimulateChainStep =
            Parent.PhaseId == 33 ||
            Parent.PhaseId == 41;

        FApplyActionResult StepResult;

        if (bSimulateChainStep)
        {
            const bool bHasPendingContext = Parent.PendingActionContext.bIsValid;

            StepResult = Internal_SimulateTransition(
                CombinedState,
                Parent.PhaseId,
                Parent.PlayerId,
                Action,
                bHasPendingContext,
                bHasPendingContext ? Parent.PendingActionContext.PendingActionA : INDEX_NONE,
                bHasPendingContext ? Parent.PendingActionContext.PendingActionB : INDEX_NONE,
                bHasPendingContext ? Parent.PendingActionContext.PendingActionC : INDEX_NONE,
                bHasPendingContext ? Parent.PendingActionContext.PendingActionD : INDEX_NONE,
                NodeIndex,
                Parent.EntityLists
            );

            Child.NodeFeatures = ExtractNodeFeatures(StepResult.OutState);
            Child.GlobalFeatures = ExtractGlobalFeatures(StepResult.OutState);
            Child.EntityLists = StepResult.OutEntityLists;
            if (Child.EntityLists.Num() != NUM_TERRITORIES)
                Child.EntityLists.SetNum(NUM_TERRITORIES);
        }
        else
        {
            Child.NodeFeatures = Parent.NodeFeatures;
            Child.GlobalFeatures = Parent.GlobalFeatures;
            Child.EntityLists = Parent.EntityLists;
        }
        Child.PhaseId = static_cast<int32>(GetNextChainPhase(ParentPhase));
        Child.PlayerId = Parent.PlayerId;
        Child.ParentIndex = NodeIndex;
        Child.ActionFromParent = Action;
        Child.bIsTerminal = false;

        const bool bParentIsSecondStep = IsSecondStepOfThreeStepChain(ParentPhase);
        const bool bParentIsThirdStep = IsThirdStepOfFourStepChain(ParentPhase);

        const EPhaseId ChildPhaseTyped = static_cast<EPhaseId>(Child.PhaseId);

        if (IsSecondStepOfThreeStepChain(ChildPhaseTyped))
        {
            // Child is second step of chain — store first action in A
            Child.PendingActionContext.PendingActionA = Action;
            Child.PendingActionContext.PendingActionB = INDEX_NONE;
            Child.PendingActionContext.PendingActionC = INDEX_NONE;
            Child.PendingActionContext.PendingActionD = INDEX_NONE;
            Child.PendingActionContext.bIsValid = true;
        }
        else if (IsThirdStepOfFourStepChain(ChildPhaseTyped))
        {
            // Child is CombatMoveUnitDest/NonCombatUnitDest.
            // Pending context was populated by the SimulateTransition path
            // (CombatMoveSource and LoadUnitsCombat calls). Carry all
            // existing pending actions forward; store destination in D.
            Child.PendingActionContext.PendingActionA =
                Parent.PendingActionContext.PendingActionA;
            Child.PendingActionContext.PendingActionB =
                Parent.PendingActionContext.PendingActionB;
            Child.PendingActionContext.PendingActionC =
                Parent.PendingActionContext.PendingActionC;
            Child.PendingActionContext.PendingActionD = INDEX_NONE;
            Child.PendingActionContext.bIsValid = true;
        }
        else if (bParentIsSecondStep)
        {
            // Parent was second step, child is chain-final of a two-step chain.
            // Carry A forward, store action in B.
            Child.PendingActionContext.PendingActionA =
                Parent.PendingActionContext.PendingActionA;
            Child.PendingActionContext.PendingActionB = Action;
            Child.PendingActionContext.PendingActionC = INDEX_NONE;
            Child.PendingActionContext.PendingActionD = INDEX_NONE;
            Child.PendingActionContext.bIsValid = true;
        }
        else if (bParentIsThirdStep)
        {
            // Parent was CombatMoveUnitDest/NonCombatUnitDest (third step).
            // Child is CombatMoveQuantity/NonCombatQuantity (chain-final).
            // Carry A/B/C (source, load1, load2) forward.
            // Store CombatMoveUnitDest action (Parent.ActionFromParent) in D.
            Child.PendingActionContext.PendingActionA =
                Parent.PendingActionContext.PendingActionA;
            Child.PendingActionContext.PendingActionB =
                Parent.PendingActionContext.PendingActionB;
            Child.PendingActionContext.PendingActionC =
                Parent.PendingActionContext.PendingActionC;
            Child.PendingActionContext.PendingActionD = Action;
            Child.PendingActionContext.bIsValid = true;
        }
        else
        {
            // First step of any chain — store action in A.
            Child.PendingActionContext.PendingActionA = Action;
            Child.PendingActionContext.PendingActionB = INDEX_NONE;
            Child.PendingActionContext.PendingActionC = INDEX_NONE;
            Child.PendingActionContext.PendingActionD = INDEX_NONE;
            Child.PendingActionContext.bIsValid = true;
        }

        if (bSimulateChainStep)
        {
            // SimulateTransition already returned the next phase's mask.
            Child.LegalActionMask = StepResult.OutLegalActionMask;
            Child.bIsTerminal = StepResult.bIsTerminal;
        }
        else
        {
            TArray<float> ChainState;
            ChainState.Reserve(Child.NodeFeatures.Num() + Child.GlobalFeatures.Num());
            ChainState.Append(Child.NodeFeatures);
            ChainState.Append(Child.GlobalFeatures);

            const FApplyActionResult MaskResult = Internal_GetLegalActionMask(
                ChainState,
                Child.PhaseId,
                Child.PlayerId,
                true,
                Child.PendingActionContext.PendingActionA,
                Child.PendingActionContext.PendingActionB,
                Child.PendingActionContext.PendingActionC,
                Child.PendingActionContext.PendingActionD,
                NodeIndex,
                Child.EntityLists
            );
            Child.LegalActionMask = MaskResult.OutLegalActionMask;
            Child.bIsTerminal = MaskResult.bIsTerminal;

        }
    }
    else
    {
        // Chain-final or standalone: call SimulateTransition.
        const bool bHasPendingContext = Parent.PendingActionContext.bIsValid;

        // For phases that start a new pending action chain, Action itself
        // is PendingActionA — not the inherited parent context.
        const bool bIsChainStartPhase =
            ParentPhase == EPhaseId::CombatMoveSource ||
            ParentPhase == EPhaseId::NonCombatSource ||
            ParentPhase == EPhaseId::StrategicBombingSource ||
            ParentPhase == EPhaseId::UnitRepair ||
            ParentPhase == EPhaseId::PurchaseType ||
            ParentPhase == EPhaseId::KamikazeQuantity ||
            ParentPhase == EPhaseId::ScrambleSource ||
            ParentPhase == EPhaseId::BombardSource ||
            ParentPhase == EPhaseId::SubmarineActionSource ||
            ParentPhase == EPhaseId::AirUnitLandOnCarrier ||
            ParentPhase == EPhaseId::PlacementCarrier ||
            ParentPhase == EPhaseId::SBR_InterceptorCommitment ||
            ParentPhase == EPhaseId::SBR_EscortCommitment;

        const int32 EffectivePendingActionA = bIsChainStartPhase
            ? Action
            : (bHasPendingContext ? Parent.PendingActionContext.PendingActionA : INDEX_NONE);
        const int32 EffectivePendingActionB =
            bHasPendingContext ? Parent.PendingActionContext.PendingActionB : INDEX_NONE;
        const int32 EffectivePendingActionC =
            bHasPendingContext ? Parent.PendingActionContext.PendingActionC : INDEX_NONE;
        const int32 EffectivePendingActionD =
            bHasPendingContext ? Parent.PendingActionContext.PendingActionD : INDEX_NONE;

        const FApplyActionResult Result = Internal_SimulateTransition(
            CombinedState,
            Parent.PhaseId,
            Parent.PlayerId,
            Action,
            bIsChainStartPhase ? true : bHasPendingContext,
            EffectivePendingActionA,
            EffectivePendingActionB,
            EffectivePendingActionC,
            EffectivePendingActionD,
            NodeIndex,
            Parent.EntityLists
        );

        Child.NodeFeatures = ExtractNodeFeatures(Result.OutState);
        Child.GlobalFeatures = ExtractGlobalFeatures(Result.OutState);
        Child.EntityLists = Result.OutEntityLists;
        Child.PhaseId = Result.OutPhaseId;
        Child.PlayerId = Result.OutPlayerId;
        Child.ParentIndex = NodeIndex;
        Child.ActionFromParent = Action;
        Child.bIsTerminal = Result.bIsTerminal;
        Child.LegalActionMask = Result.OutLegalActionMask;

        if (Child.EntityLists.Num() != NUM_TERRITORIES)
            Child.EntityLists.SetNum(NUM_TERRITORIES);

        ensureMsgf(
            Child.PhaseId >= 0,
            TEXT("Invalid PhaseId produced by SimulateTransition for NodeIndex %d"),
            NodeIndex);

        // ----------------------------------------------------------------
        // PENDING ACTION CONTEXT — carried forward for multi-step
        // SimulateTransition chains (CombatMoveSource, LoadUnitsCombat).
        //
        // CombatMoveSource (parent) → LoadUnitsCombat (child):
        //   PendingActionA = CombatMoveSource action
        //
        // LoadUnitsCombat first call (parent) → LoadUnitsCombat second call (child):
        //   PendingActionA carried, PendingActionB = first load action
        //
        // LoadUnitsCombat second call (parent) → CombatMoveUnitDest (child):
        //   PendingActionA/B carried, PendingActionC = second load action
        //
        // LoadUnitsCombat single call (parent) → CombatMoveUnitDest (child):
        //   PendingActionA carried, PendingActionB = load action
        //
        // All other SimulateTransition phases: reset context.
        // ----------------------------------------------------------------
        const EPhaseId ResultPhase = static_cast<EPhaseId>(Child.PhaseId);

        if (ParentPhase == EPhaseId::CombatMoveSource ||
            ParentPhase == EPhaseId::NonCombatSource ||
            ParentPhase == EPhaseId::StrategicBombingSource ||
            ParentPhase == EPhaseId::UnitRepair ||
            ParentPhase == EPhaseId::PurchaseType ||
            ParentPhase == EPhaseId::KamikazeQuantity ||
            ParentPhase == EPhaseId::ScrambleSource ||
            ParentPhase == EPhaseId::BombardSource ||
            ParentPhase == EPhaseId::SubmarineActionSource ||
            ParentPhase == EPhaseId::AirUnitLandOnCarrier ||
            ParentPhase == EPhaseId::PlacementCarrier ||
            ParentPhase == EPhaseId::SBR_InterceptorCommitment ||
            ParentPhase == EPhaseId::SBR_EscortCommitment)
        {
            // SimulateTransition phases that store action in A for chain-final child.
            // CombatMoveSource/NonCombatSource/StrategicBombingSource carry forward
            // into multi-step chains. UnitRepair/PurchaseType/SBR commitment phases
            // store action in A so quantity phases can read it as PendingActionA.
            Child.PendingActionContext.PendingActionA = Action;
            Child.PendingActionContext.PendingActionB = INDEX_NONE;
            Child.PendingActionContext.PendingActionC = INDEX_NONE;
            Child.PendingActionContext.PendingActionD = INDEX_NONE;
            Child.PendingActionContext.bIsValid = true;
        }
        else if ((ParentPhase == EPhaseId::LoadUnitsCombat ||
            ParentPhase == EPhaseId::LoadUnitsNonCombat) &&
            (ResultPhase == EPhaseId::LoadUnitsCombat ||
                ResultPhase == EPhaseId::LoadUnitsNonCombat))
        {
            // First LoadUnitsCombat fired, Blueprint returned another LoadUnitsCombat.
            // Carry A forward, store first load action in B.
            Child.PendingActionContext.PendingActionA =
                Parent.PendingActionContext.PendingActionA;
            Child.PendingActionContext.PendingActionB = Action;
            Child.PendingActionContext.PendingActionC = INDEX_NONE;
            Child.PendingActionContext.PendingActionD = INDEX_NONE;
            Child.PendingActionContext.bIsValid = true;
        }
        else if ((ParentPhase == EPhaseId::LoadUnitsCombat ||
            ParentPhase == EPhaseId::LoadUnitsNonCombat) &&
            (ResultPhase == EPhaseId::CombatMoveUnitDest ||
                ResultPhase == EPhaseId::NonCombatUnitDest))
        {
            // LoadUnitsCombat fired and Blueprint returned CombatMoveUnitDest.
            // This is either the first (single) or second load call.
            // If PendingActionB is already set, this is the second load:
            //   carry A/B forward, store second load action in C.
            // If PendingActionB is INDEX_NONE, this is a single load:
            //   carry A forward, store load action in B.
            Child.PendingActionContext.PendingActionA =
                Parent.PendingActionContext.PendingActionA;
            if (Parent.PendingActionContext.PendingActionB != INDEX_NONE)
            {
                Child.PendingActionContext.PendingActionB =
                    Parent.PendingActionContext.PendingActionB;
                Child.PendingActionContext.PendingActionC = Action;
            }
            else
            {
                Child.PendingActionContext.PendingActionB = Action;
                Child.PendingActionContext.PendingActionC = INDEX_NONE;
            }
            Child.PendingActionContext.PendingActionD = INDEX_NONE;
            Child.PendingActionContext.bIsValid = true;
        }
        else if (ParentPhase == EPhaseId::PlacementQuantity &&
            ResultPhase == EPhaseId::AirUnitLandOnCarrierCount)
        {
            // PlacementCarrier -> PlacementQuantity -> AirUnitLandOnCarrierCount.
            // Phase 45 needs A = the PlacementCarrier action (which carrier and
            // sea zone the plane is assigned to). Carry A forward.
            Child.PendingActionContext.PendingActionA =
                Parent.PendingActionContext.bIsValid
                ? Parent.PendingActionContext.PendingActionA
                : INDEX_NONE;
            Child.PendingActionContext.PendingActionB = INDEX_NONE;
            Child.PendingActionContext.PendingActionC = INDEX_NONE;
            Child.PendingActionContext.PendingActionD = INDEX_NONE;
            Child.PendingActionContext.bIsValid = Parent.PendingActionContext.bIsValid;
        }
        else
        {
            // All other SimulateTransition phases — reset context.
            Child.PendingActionContext = FPendingActionContext();
        }
    }

    const int32 ActionSpaceSize =
        GetPolicySizeForPhase(static_cast<EPhaseId>(Child.PhaseId));
    Child.PolicyPrior.Init(0.0f, ActionSpaceSize);
    Child.EdgeVisitCount.Init(0, ActionSpaceSize);
    Child.VirtualLossEdgeCount.Init(0, ActionSpaceSize);
    Child.EdgeValueSum.Init(0.0f, ActionSpaceSize);
    Child.VisitCount = 0;
    Child.TotalValue = 0.0f;
    Child.bIsExpanded = false;
    Child.bPolicyInitialized = false;
    Child.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);

    // ----------------------------------------------------------------
    // PENDING-ACTION FEATURES: GlobalFeatures[147..150] = A, B, C, D
    // Each value is (action + 1) / action-space size of the phase that
    // chose it, so 0 means "no pending action". A value carried over
    // from the parent keeps the parent's feature; a value chosen at the
    // parent phase is encoded with the parent phase's action-space size.
    // (Global feature 551 is left to Blueprint.)
    // ----------------------------------------------------------------
    {
        constexpr int32 GLOBAL_PENDING_FIRST = 147;   // A=147, B=148, C=149, D=150

        if (Child.GlobalFeatures.IsValidIndex(GLOBAL_PENDING_FIRST + 3))
        {
            const FPendingActionContext& CC = Child.PendingActionContext;
            const FPendingActionContext& PC = Parent.PendingActionContext;

            const int32 ChildSlots[4] = {
                CC.PendingActionA, CC.PendingActionB, CC.PendingActionC, CC.PendingActionD };
            const int32 ParentSlots[4] = {
                PC.PendingActionA, PC.PendingActionB, PC.PendingActionC, PC.PendingActionD };

            const int32 ParentActionSize = GetPolicySizeForPhase(ParentPhase);

            for (int32 k = 0; k < 4; k++)
            {
                const int32 FeatureIndex = GLOBAL_PENDING_FIRST + k;
                float Value = 0.0f;

                if (CC.bIsValid && ChildSlots[k] >= 0)
                {
                    const bool bCarried =
                        PC.bIsValid &&
                        ParentSlots[k] == ChildSlots[k] &&
                        Parent.GlobalFeatures.IsValidIndex(FeatureIndex);

                    if (bCarried)
                        Value = Parent.GlobalFeatures[FeatureIndex];
                    else if (ParentActionSize > 0)
                        Value = FMath::Clamp(
                            (float)(ChildSlots[k] + 1) / (float)ParentActionSize, 0.0f, 1.0f);
                }

                Child.GlobalFeatures[FeatureIndex] = Value;
            }
        }
    }

    // CRITICAL: re-fetch parent after CreateNode to avoid stale reference
    ChildIndex = Tree.CreateNode(Child);
    Tree.GetNode(NodeIndex).ChildIndices.Add(ChildIndex);

    FMCTSNode& CreatedChild = Tree.GetNode(ChildIndex);

    // ----------------------------------------------------------------
    // DICE SAMPLING DISPATCH
    // Triggered by the parent phase that just resolved (for tech,
    // kamikaze, bombardment, strategic bombing, submarine action) or
    // by the child's incoming phase (for combat casualty resolution).
    // All sampling writes results into CreatedChild's GlobalFeatures
    // before EvaluateNode is called so the model sees the outcome.
    // ParentPhase and ParentPlayerId are captured before CreateNode
    // since the Parent reference may be invalidated by reallocation.
    //
    // Battle state is now in GlobalFeatures[512-547].
    // Hits pending are read from GLOBAL_BATTLE_ATK_HITS/DEF_HITS.
    // No node feature scan is performed.
    // ----------------------------------------------------------------
    const EPhaseId ChildPhase = static_cast<EPhaseId>(CreatedChild.PhaseId);

    if (ChildPhase == EPhaseId::CombatResolveCasualtyType ||
        ChildPhase == EPhaseId::SBR_AirBattleCasualtyType)
    {
        // Combat dice or SBR air battle dice.
        // Check global battle state for pending hits before rolling
        // to avoid double-rolling when hits were pre-populated by
        // a prior bombardment, kamikaze, or sub surprise sampling call.
        const float AtkPending =
            CreatedChild.GlobalFeatures.IsValidIndex(GLOBAL_BATTLE_ATK_HITS)
            ? CreatedChild.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] : 0.0f;
        const float DefPending =
            CreatedChild.GlobalFeatures.IsValidIndex(GLOBAL_BATTLE_DEF_HITS)
            ? CreatedChild.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] : 0.0f;
        const bool bHitsAlreadyPending =
            (AtkPending > 1e-5f) || (DefPending > 1e-5f);

        if (!bHitsAlreadyPending)
        {
            const bool bDiceRolled = SampleCombatDice(ChildIndex);
            ensureMsgf(bDiceRolled,
                TEXT("ExpandNode: SampleCombatDice failed for child %d"),
                ChildIndex);

            if (bDiceRolled)
            {
                TArray<float> CasualtyState;
                CasualtyState.Reserve(
                    CreatedChild.NodeFeatures.Num() +
                    CreatedChild.GlobalFeatures.Num());
                CasualtyState.Append(CreatedChild.NodeFeatures);
                CasualtyState.Append(CreatedChild.GlobalFeatures);

                const FApplyActionResult CasualtyMaskResult =
                    Internal_GetCasualtyAssignmentMask(
                        CasualtyState,
                        CreatedChild.PhaseId,
                        CreatedChild.PlayerId,
                        CreatedChild.EntityLists);

                const int32 CreatedChildActionSize =
                    GetPolicySizeForPhase(
                        static_cast<EPhaseId>(CreatedChild.PhaseId));
                if (CasualtyMaskResult.OutLegalActionMask.Num() ==
                    CreatedChildActionSize)
                {
                    CreatedChild.LegalActionMask =
                        CasualtyMaskResult.OutLegalActionMask;
                }
            }
        }
    }
    else if (ParentPhase == EPhaseId::Tech)
    {
        // Tech dice: Action is number of dice purchased (0-6).
        // Action 0 = skip tech purchase — no dice to roll.
        if (Action > 0)
            SampleTechDice(ChildIndex, Action, ParentPlayerId);
    }
    else if (ParentPhase == EPhaseId::KamikazeTarget)
    {
        // Kamikaze strikes: quantity and location in PendingActionContext.
        // Hits written to GlobalFeatures[GLOBAL_BATTLE_DEF_HITS].
        // Also decrements GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING].
        SampleKamikazeDice(ChildIndex);
    }
    else if (ParentPhase == EPhaseId::BombardQuantity)
    {
        // Offshore bombardment: battleship/cruiser dice after quantity committed.
        // Hits written to GlobalFeatures[GLOBAL_BATTLE_ATK_HITS].
        SampleBombardmentDice(ChildIndex);
    }
    else if (ParentPhase == EPhaseId::StrategicBombingQuantity)
    {
        // SBR: bomber dice after quantity committed.
        // Facility damage written to NodeFeatures[territory*20+5].
        SampleStrategicBombingDice(ChildIndex);
    }
    else if (ParentPhase == EPhaseId::SubmarineActionQuantity)
    {
        // Submarine surprise strike: dice after quantity committed.
        // Hits written to GlobalFeatures[GLOBAL_BATTLE_ATK_HITS/DEF_HITS].
        SampleSubmarineSurpriseDice(ChildIndex);
    }

    // Terminal nodes do not need inference — outcome values are in global features
    if (CreatedChild.bIsTerminal)
        return;

    // Request network evaluation (uses the inference cache when possible)
    EvaluateNode(ChildIndex);
}


// ----------------------------------------------------------------
// Search timing (for benchmarking runtimes and simulation counts).
// Set bLogSearchTiming to false to silence the per-search summary.
// Counters are reset at the start of each GetActionFromMCTS search.
// Place this block and MakeInferenceKey above EvaluateNode,
// FlushInferenceBatch and GetActionFromMCTS in UAIManager.cpp.
// ----------------------------------------------------------------
namespace MCTSTiming
{
    static constexpr bool bLogSearchTiming = false;
    static double InferenceSeconds = 0.0;
    static int32  InferenceCalls = 0;
    static int32  InferenceSamples = 0;

    static void Reset()
    {
        InferenceSeconds = 0.0;
        InferenceCalls = 0;
        InferenceSamples = 0;
    }
}

// ----------------------------------------------------------------
// Inference cache key: hashes exactly what the network sees (node,
// global and entity inputs) plus the phase/player/action size that
// select the policy head. Two requests with the same key produce the
// same network output, so one result can serve both.
// Place this above EvaluateNode (it is also used by FlushInferenceBatch).
// ----------------------------------------------------------------
static FInferenceCacheKey MakeInferenceKey(const FInferenceRequest& Req)
{
    uint32 Hash = 0;
    auto Mix = [&Hash](const TArray<float>& Data)
        {
            const uint32 H = Data.Num() > 0
                ? FCrc::MemCrc32(Data.GetData(), Data.Num() * sizeof(float))
                : 0u;
            Hash = HashCombine(Hash, H);
        };
    Mix(Req.NodeFeatures);
    Mix(Req.GlobalFeatures);
    Mix(Req.EntityTensor);
    Mix(Req.EntityCounts);

    FInferenceCacheKey Key;
    Key.StateHash = Hash;
    Key.PhaseId = Req.PhaseId;
    Key.PlayerId = Req.PlayerId;
    Key.ActionSize = Req.ActionSize;
    return Key;
}

void UAIManager::EvaluateNode(int32 NodeIndex)
{
    if (!InferenceQueue || !Tree.Nodes.IsValidIndex(NodeIndex))
        return;

    if (InFlightNodes.Contains(NodeIndex))
        return;

    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    ValidatePhaseRoutingInvariant(NodeIndex);

    if (Node.bPolicyInitialized)
        return;

    if (Node.bIsTerminal)
    {
        Node.bPolicyInitialized = false;
        Node.bIsExpanded = false;
        Node.PolicyPrior.Reset();
        Node.EdgeVisitCount.Reset();
        Node.EdgeValueSum.Reset();
        return;
    }

    const int32 ActionSize = GetActionSizeForPhase(Node.PhaseId);
    if (ActionSize <= 0)
        return;

    if (Node.PolicyPrior.Num() != ActionSize) Node.PolicyPrior.Init(0.0f, ActionSize);
    if (Node.EdgeVisitCount.Num() != ActionSize) Node.EdgeVisitCount.Init(0, ActionSize);
    if (Node.EdgeValueSum.Num() != ActionSize) Node.EdgeValueSum.Init(0.0f, ActionSize);

    if (Node.LegalActionMask.Num() != ActionSize)
    {
        ensureMsgf(false,
            TEXT("LegalActionMask missing or invalid size for NodeIndex %d"), NodeIndex);
        return;
    }

    // ---- Build the request (the key includes the entity inputs) ----
    FInferenceRequest Req;
    Req.NodeFeatures = Node.NodeFeatures;
    Req.GlobalFeatures = Node.GlobalFeatures;
    AssembleEntityTensor(Node.EntityLists, Req.EntityTensor, Req.EntityCounts);
    Req.PhaseId = Node.PhaseId;
    Req.PlayerId = Node.PlayerId;
    Req.NodeIndex = NodeIndex;
    Req.ActionSize = ActionSize;

    // ---- Cache hit: apply immediately ----
    const FInferenceCacheKey Key = MakeInferenceKey(Req);
    TPair<TArray<float>, TArray<float>> Cached;
    if (InferenceQueue->TryGetCache(Key, Cached))
    {
        ApplyInferenceResultToNode(NodeIndex, Cached.Key, Cached.Value);
        return;
    }

    // ---- Otherwise queue it; FlushInferenceBatch answers every request ----
    InFlightNodes.Add(NodeIndex);
    InferenceQueue->Enqueue(Req);
}

int32 UAIManager::SelectActionPUCT(int32 NodeIndex, float C)
{
    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    const int32 ActionSize =
        GetPolicySizeForPhase(static_cast<EPhaseId>(Node.PhaseId));
    if (ActionSize <= 0)
        return -1;

    const float VisitCountF =
        (float)(Node.VisitCount + Node.VirtualLossCount);
    const float SqrtN =
        (VisitCountF > 0.0f) ? FMath::Sqrt(VisitCountF) : 1.0f;

    bool bHasAnyLegal = false;
    for (int32 Action = 0; Action < ActionSize; Action++)
    {
        if (Node.LegalActionMask.IsValidIndex(Action) &&
            Node.LegalActionMask[Action])
        {
            bHasAnyLegal = true;
            break;
        }
    }
    if (!bHasAnyLegal)
        return -1;
    if (!bHasAnyLegal)
        return -1;

    float BestScore = -FLT_MAX;
    int32 BestAction = -1;

    for (int32 Action = 0; Action < ActionSize; Action++)
    {
        const bool bLegal =
            Node.LegalActionMask.IsValidIndex(Action) &&
            Node.LegalActionMask[Action];
        if (!bLegal)
            continue;

        const float Prior =
            Node.PolicyPrior.IsValidIndex(Action)
            ? FMath::Clamp(Node.PolicyPrior[Action], 1e-8f, 1.0f)
            : 1e-8f;

        const int32 EdgeVisits =
            Node.EdgeVisitCount.IsValidIndex(Action)
            ? Node.EdgeVisitCount[Action]
            : 0;

        const float EdgeValue =
            Node.EdgeValueSum.IsValidIndex(Action)
            ? Node.EdgeValueSum[Action]
            : 0.0f;

        const int32 VirtualVisits =
            Node.VirtualLossEdgeCount.IsValidIndex(Action)
            ? Node.VirtualLossEdgeCount[Action]
            : 0;

        const int32 TotalVisits = EdgeVisits + VirtualVisits;
        const float TotalValue = EdgeValue - (float)VirtualVisits;
        const float Q =
            (TotalVisits > 0)
            ? (TotalValue / (float)TotalVisits)
            : 0.0f;

        const float U =
            C * Prior * (SqrtN / (1.0f + (float)TotalVisits));
        const float Score = Q + U;

        if (Score > BestScore)
        {
            BestScore = Score;
            BestAction = Action;
        }
    }

    return BestAction;
}

void UAIManager::Backpropagate(int32 NodeIndex, int32 RootPlayerId)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
        return;

    // ----------------------------------------------------------------
    // LEAF VALUE — one value per player.
    // The value of the node this simulation reached is propagated up
    // the whole path. Terminal nodes use the game outcome Blueprint
    // wrote into GlobalFeatures[GLOBAL_OUTCOME_VALUES + p]; all other
    // nodes use the network's value estimate.
    // ----------------------------------------------------------------
    float LeafValues[NUM_PLAYERS];
    {
        const FMCTSNode& Leaf = Tree.GetNode(NodeIndex);
        for (int32 p = 0; p < NUM_PLAYERS; p++)
        {
            float V = 0.0f;
            if (Leaf.bIsTerminal)
            {
                const int32 OutcomeIndex = GLOBAL_OUTCOME_VALUES + p;
                if (Leaf.GlobalFeatures.IsValidIndex(OutcomeIndex))
                    V = Leaf.GlobalFeatures[OutcomeIndex];
            }
            else if (Leaf.MCTSValuePerPlayer.IsValidIndex(p))
            {
                V = Leaf.MCTSValuePerPlayer[p];
            }
            LeafValues[p] = FMath::IsFinite(V) ? V : 0.0f;
        }
    }

    auto ValueFor = [&LeafValues](int32 PlayerId) -> float
        {
            return LeafValues[FMath::Clamp(PlayerId, 0, NUM_PLAYERS - 1)];
        };

    // RootPlayerId is no longer needed: every edge is scored from the
    // view of the player who chooses that edge (the parent's player).

    int32 CurrentNodeIndex = NodeIndex;
    while (CurrentNodeIndex != INDEX_NONE &&
        Tree.Nodes.IsValidIndex(CurrentNodeIndex))
    {
        FMCTSNode& Node = Tree.GetNode(CurrentNodeIndex);

        // ----------------------------------------------------------------
        // NODE STATISTICS UPDATE
        // ----------------------------------------------------------------
        Node.VisitCount += 1;
        Node.TotalValue += ValueFor(Node.PlayerId);

        // Remove virtual loss for this node
        Node.VirtualLossCount = FMath::Max(0, Node.VirtualLossCount - 1);

        const int32 ParentIndex = Node.ParentIndex;
        if (ParentIndex != INDEX_NONE &&
            Tree.Nodes.IsValidIndex(ParentIndex))
        {
            FMCTSNode& Parent = Tree.GetNode(ParentIndex);
            const int32 Action = Node.ActionFromParent;

            // Edge value from the view of the player acting at the parent.
            const float EdgeValue = ValueFor(Parent.PlayerId);

            if (Parent.EdgeVisitCount.IsValidIndex(Action))
            {
                Parent.EdgeVisitCount[Action] += 1;
            }
            else if (Action >= 0)
            {
                Parent.EdgeVisitCount.SetNum(
                    FMath::Max(Parent.EdgeVisitCount.Num(), Action + 1));
                Parent.EdgeVisitCount[Action] = 1;
            }

            if (Parent.EdgeValueSum.IsValidIndex(Action))
            {
                Parent.EdgeValueSum[Action] += EdgeValue;
            }
            else if (Action >= 0)
            {
                Parent.EdgeValueSum.SetNum(
                    FMath::Max(Parent.EdgeValueSum.Num(), Action + 1));
                Parent.EdgeValueSum[Action] = EdgeValue;
            }

            // Remove virtual loss edge count on parent
            if (Parent.VirtualLossEdgeCount.IsValidIndex(Action))
                Parent.VirtualLossEdgeCount[Action] =
                FMath::Max(0, Parent.VirtualLossEdgeCount[Action] - 1);
        }

        CurrentNodeIndex = Node.ParentIndex;
    }
}

void UAIManager::ApplyInferenceResultToNode(
    int32 NodeIndex,
    const TArray<float>& PolicyOutput,
    const TArray<float>& ValueOutput)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
    {
        return;
    }

    FMCTSNode& Node = Tree.GetNode(NodeIndex);

    const int32 ActionSize =
        GetActionSizeForPhase(Node.PhaseId);

    if (ActionSize <= 0)
    {
        return;
    }

    // -----------------------------
    // POLICY INITIALIZATION — SOFTMAX OVER LEGAL ACTIONS ONLY
    // The raw model output is logits. We apply softmax masked to legal
    // actions so PolicyPrior is always a valid probability distribution
    // before Dirichlet noise or PUCT scoring is applied.
    // Illegal actions receive 0.0 probability.
    // -----------------------------
    const int32 PolicySize = FMath::Min(ActionSize, PolicyOutput.Num());

    if (Node.PolicyPrior.Num() != ActionSize)
    {
        Node.PolicyPrior.Init(0.0f, ActionSize);
    }

    // Find max logit over legal actions for numerical stability
    float MaxLogit = -FLT_MAX;
    for (int32 i = 0; i < PolicySize; i++)
    {
        const bool bLegal =
            Node.LegalActionMask.IsValidIndex(i) && Node.LegalActionMask[i];
        if (bLegal && PolicyOutput[i] > MaxLogit)
            MaxLogit = PolicyOutput[i];
    }

    if (MaxLogit <= -FLT_MAX)
        MaxLogit = 0.0f;

    // Compute exp(logit - max) for legal actions; zero for illegal
    float ExpSum = 0.0f;
    for (int32 i = 0; i < PolicySize; i++)
    {
        const bool bLegal =
            Node.LegalActionMask.IsValidIndex(i) && Node.LegalActionMask[i];
        if (bLegal)
        {
            const float E = FMath::Exp(PolicyOutput[i] - MaxLogit);
            Node.PolicyPrior[i] = E;
            ExpSum += E;
        }
        else
        {
            Node.PolicyPrior[i] = 0.0f;
        }
    }

    // Normalize to produce a valid probability distribution
    if (ExpSum > 1e-8f)
    {
        const float InvExpSum = 1.0f / ExpSum;
        for (int32 i = 0; i < PolicySize; i++)
        {
            Node.PolicyPrior[i] *= InvExpSum;
        }
    }
    else
    {
        // Fallback: uniform over legal actions if all logits collapse
        int32 LegalCount = 0;
        for (int32 i = 0; i < ActionSize; i++)
        {
            if (Node.LegalActionMask.IsValidIndex(i) && Node.LegalActionMask[i])
                LegalCount++;
        }
        const float UniformP = (LegalCount > 0) ? (1.0f / (float)LegalCount) : 0.0f;
        for (int32 i = 0; i < ActionSize; i++)
        {
            const bool bLegal =
                Node.LegalActionMask.IsValidIndex(i) && Node.LegalActionMask[i];
            Node.PolicyPrior[i] = bLegal ? UniformP : 0.0f;
        }
    }

    // -----------------------------
    // VALUE INITIALIZATION (NETWORK ONLY)
    // -----------------------------
    const int32 ValueSize = ValueOutput.Num();

    if (Node.MCTSValuePerPlayer.Num() != ValueSize)
    {
        Node.MCTSValuePerPlayer.Init(0.0f, ValueSize);
    }

    for (int32 i = 0; i < ValueSize; i++)
    {
        Node.MCTSValuePerPlayer[i] = ValueOutput[i];
    }

    // -----------------------------
    // MCTS STATE FLAGS
    // -----------------------------
    Node.bPolicyInitialized = true;
    Node.bIsExpanded = true;

    // -----------------------------
    // CRITICAL FIX: RELEASE INFERENCE LOCK
    // -----------------------------
    InFlightNodes.Remove(NodeIndex);
}

// FlushInferenceBatch
// - Requests with identical inputs (same key) are run once and the
//   result is applied to every node that asked for it.
// - All phases share one batch: the network computes every head in a
//   single pass, and each node reads its own phase's head.
// - Results are stored in the inference cache.
// ----------------------------------------------------------------
void UAIManager::FlushInferenceBatch()
{
    if (!InferenceQueue || !RDGCache)
        return;

    TArray<FInferenceRequest> Requests;
    {
        static FCriticalSection FlushMutex;
        FScopeLock Lock(&FlushMutex);
        InferenceQueue->DequeueAll(Requests);
    }
    if (Requests.Num() == 0)
        return;

    // ---- Group identical requests ----
    TArray<FInferenceRequest>   Unique;
    TArray<FInferenceCacheKey>  UniqueKeys;
    TArray<TArray<int32>>       NodesPerUnique;
    TMap<FInferenceCacheKey, int32> KeyToUnique;

    for (FInferenceRequest& Req : Requests)
    {
        const FInferenceCacheKey Key = MakeInferenceKey(Req);

        // Already answered (e.g. by an earlier flush in this search)
        TPair<TArray<float>, TArray<float>> Cached;
        if (InferenceQueue->TryGetCache(Key, Cached))
        {
            ApplyInferenceResultToNode(Req.NodeIndex, Cached.Key, Cached.Value);
            continue;
        }

        if (const int32* Existing = KeyToUnique.Find(Key))
        {
            NodesPerUnique[*Existing].Add(Req.NodeIndex);
            continue;
        }

        const int32 NewIdx = Unique.Num();
        KeyToUnique.Add(Key, NewIdx);
        UniqueKeys.Add(Key);
        NodesPerUnique.Add({ Req.NodeIndex });
        Unique.Add(MoveTemp(Req));
    }

    if (Unique.Num() == 0)
        return;

    static FCriticalSection TreeMutationMutex;
    FScopeLock TreeLock(&TreeMutationMutex);

    // ---- Run in fixed-size batches ----
    // Every call uses the same batch size, so the runtime never has to
    // re-prepare the model for a new input shape. Short batches are padded
    // by repeating their last request; padded results are ignored.
    // Equal to the batch size the ONNX model was exported with.
    constexpr int32 FixedBatchSize = INFERENCE_BATCH_SIZE;

    for (int32 Start = 0; Start < Unique.Num(); Start += FixedBatchSize)
    {
        const int32 Count = FMath::Min(FixedBatchSize, Unique.Num() - Start);

        TArray<TArray<float>> NodeFeaturesBatch;
        TArray<TArray<float>> GlobalFeaturesBatch;
        TArray<TArray<float>> EntityTensorBatch;
        TArray<TArray<float>> EntityCountsBatch;
        TArray<float>         PhaseIds;
        NodeFeaturesBatch.Reserve(FixedBatchSize);
        GlobalFeaturesBatch.Reserve(FixedBatchSize);
        EntityTensorBatch.Reserve(FixedBatchSize);
        EntityCountsBatch.Reserve(FixedBatchSize);
        PhaseIds.Reserve(FixedBatchSize);

        for (int32 i = 0; i < FixedBatchSize; i++)
        {
            // i >= Count: padding (repeat the last real request)
            const FInferenceRequest& Req = Unique[Start + FMath::Min(i, Count - 1)];
            NodeFeaturesBatch.Add(Req.NodeFeatures);
            GlobalFeaturesBatch.Add(Req.GlobalFeatures);
            EntityTensorBatch.Add(Req.EntityTensor);
            EntityCountsBatch.Add(Req.EntityCounts);
            PhaseIds.Add((float)Req.PhaseId);
        }

        TArray<float> H2, H3, H7, H10, H13, H14, H20, H49;
        TArray<float> H128, H202, H330, H6581, H988, Value;

        const double InferenceStart = FPlatformTime::Seconds();
        const bool bRan =
            RDGCache->RunInference(
                NodeFeaturesBatch, GlobalFeaturesBatch,
                EntityTensorBatch, EntityCountsBatch, PhaseIds,
                H2, H3, H7, H10, H13, H14, H20, H49,
                H128, H202, H330, H6581, H988, Value);
        MCTSTiming::InferenceSeconds += FPlatformTime::Seconds() - InferenceStart;
        MCTSTiming::InferenceCalls++;
        MCTSTiming::InferenceSamples += Count;

        const bool bOk =
            bRan &&
            ValidateInferenceOutputsNumerical(
                FixedBatchSize,
                H2, H3, H7, H10, H13, H14, H20, H49,
                H128, H202, H330, H6581, H988, Value);

        if (!bOk)
        {
            UE_LOG(LogTemp, Error,
                TEXT("FlushInferenceBatch: inference failed for %d requests"), Count);
            // Release the nodes so they can be requested again.
            for (int32 i = 0; i < Count; i++)
                for (int32 NodeIdx : NodesPerUnique[Start + i])
                    InFlightNodes.Remove(NodeIdx);
            continue;
        }

        for (int32 i = 0; i < Count; i++)
        {
            const FInferenceRequest& Req = Unique[Start + i];
            const EPolicyHead Head =
                GetPolicyHeadForPhase(static_cast<EPhaseId>(Req.PhaseId));

            const TArray<float>* Source = nullptr;
            switch (Head)
            {
            case EPolicyHead::Head2:    Source = &H2;    break;
            case EPolicyHead::Head3:    Source = &H3;    break;
            case EPolicyHead::Head7:    Source = &H7;    break;
            case EPolicyHead::Head10:   Source = &H10;   break;
            case EPolicyHead::Head13:   Source = &H13;   break;
            case EPolicyHead::Head14:   Source = &H14;   break;
            case EPolicyHead::Head20:   Source = &H20;   break;
            case EPolicyHead::Head49:   Source = &H49;   break;
            case EPolicyHead::Head128:  Source = &H128;  break;
            case EPolicyHead::Head202:  Source = &H202;  break;
            case EPolicyHead::Head330:  Source = &H330;  break;
            case EPolicyHead::Head6581: Source = &H6581; break;
            case EPolicyHead::Head988:  Source = &H988;  break;
            default: break;
            }

            const int32 ActionSize = GetPolicySizeForHead(Head);
            if (!Source || ActionSize <= 0 ||
                (i + 1) * ActionSize > Source->Num() ||
                (i + 1) * NUM_PLAYERS > Value.Num())
            {
                ensureMsgf(false,
                    TEXT("FlushInferenceBatch: bad policy/value slice for phase %d"),
                    Req.PhaseId);
                for (int32 NodeIdx : NodesPerUnique[Start + i])
                    InFlightNodes.Remove(NodeIdx);
                continue;
            }

            TArray<float> PolicySlice;
            PolicySlice.SetNumUninitialized(ActionSize);
            FMemory::Memcpy(PolicySlice.GetData(),
                Source->GetData() + i * ActionSize, ActionSize * sizeof(float));

            TArray<float> ValueSlice;
            ValueSlice.SetNumUninitialized(NUM_PLAYERS);
            FMemory::Memcpy(ValueSlice.GetData(),
                Value.GetData() + i * NUM_PLAYERS, NUM_PLAYERS * sizeof(float));

            InferenceQueue->StoreCache(UniqueKeys[Start + i], PolicySlice, ValueSlice);

            for (int32 NodeIdx : NodesPerUnique[Start + i])
                ApplyInferenceResultToNode(NodeIdx, PolicySlice, ValueSlice);
        }
    }
}

int32 UAIManager::SelectFinalActionFromVisits(int32 RootIndex, float Temperature)
{
    FMCTSNode& Root = Tree.GetNode(RootIndex);

    const int32 ActionSize = Root.PolicyPrior.Num();

    if (ActionSize <= 0)
    {
        return -1;
    }

    // -----------------------------
    // LEGAL ACTION AUTHORITY CHECK
    // -----------------------------
    bool bHasLegal = false;
    for (int32 i = 0; i < ActionSize; i++)
    {
        if (Root.LegalActionMask.IsValidIndex(i) && Root.LegalActionMask[i])
        {
            bHasLegal = true;
            break;
        }
    }

    if (!bHasLegal)
    {
        return -1;
    }

    TArray<float> VisitCounts;
    VisitCounts.Init(0.0f, ActionSize);

    float Sum = 0.0f;

    // -----------------------------
    // STRICT VISIT EXTRACTION (NO SEMANTICS)
    // -----------------------------
    for (int32 Action = 0; Action < ActionSize; Action++)
    {
        const bool bLegal =
            Root.LegalActionMask.IsValidIndex(Action) &&
            Root.LegalActionMask[Action];

        if (!bLegal)
        {
            continue;
        }

        const float Visits =
            Root.EdgeVisitCount.IsValidIndex(Action)
            ? (float)Root.EdgeVisitCount[Action]
            : 0.0f;

        VisitCounts[Action] = FMath::Max(0.0f, Visits);
        Sum += VisitCounts[Action];
    }

    // -----------------------------
    // FALLBACK: UNIFORM OVER LEGAL ACTIONS ONLY
    // -----------------------------
    if (Sum <= 1e-8f)
    {
        int32 LegalCount = 0;

        for (int32 i = 0; i < ActionSize; i++)
        {
            if (Root.LegalActionMask.IsValidIndex(i) && Root.LegalActionMask[i])
            {
                LegalCount++;
            }
        }

        if (LegalCount == 0)
        {
            return -1;
        }

        float InvLegal = 1.0f / (float)LegalCount;

        TArray<float> Prob;
        Prob.Init(0.0f, ActionSize);

        for (int32 i = 0; i < ActionSize; i++)
        {
            if (Root.LegalActionMask.IsValidIndex(i) && Root.LegalActionMask[i])
            {
                Prob[i] = InvLegal;
            }
        }

        float R = MCTSRandStream.FRand();
        float Acc = 0.0f;

        for (int32 i = 0; i < ActionSize; i++)
        {
            Acc += Prob[i];
            if (R <= Acc)
            {
                return i;
            }
        }

        return ActionSize - 1;
    }

    // -----------------------------
    // TEMPERATURE ~0: PLAY THE MOST-VISITED LEGAL ACTION
    // -----------------------------
    if (Temperature <= 1e-6f)
    {
        int32 BestAction = -1;
        float BestVisits = -1.0f;
        for (int32 Action = 0; Action < ActionSize; Action++)
        {
            const bool bLegal =
                Root.LegalActionMask.IsValidIndex(Action) &&
                Root.LegalActionMask[Action];
            if (bLegal && VisitCounts[Action] > BestVisits)
            {
                BestVisits = VisitCounts[Action];
                BestAction = Action;
            }
        }
        return BestAction;
    }

    // -----------------------------
    // TEMPERATURE-ONLY FINAL POLICY TRANSFORM
    // (NO PUCT OR TREE POLICY IMPACT)
    // -----------------------------
    TArray<float> Prob;
    Prob.Init(0.0f, ActionSize);

    const float InvTemp = 1.0f / Temperature;

    float Norm = 0.0f;

    for (int32 Action = 0; Action < ActionSize; Action++)
    {
        const bool bLegal =
            Root.LegalActionMask.IsValidIndex(Action) &&
            Root.LegalActionMask[Action];

        if (!bLegal)
        {
            continue;
        }

        const float V = VisitCounts[Action];

        const float Powered =
            (V > 0.0f)
            ? FMath::Pow(V, InvTemp)
            : 0.0f;

        Prob[Action] = Powered;
        Norm += Powered;
    }

    if (Norm <= 1e-8f)
    {
        int32 LegalCount = 0;

        for (int32 i = 0; i < ActionSize; i++)
        {
            if (Root.LegalActionMask.IsValidIndex(i) && Root.LegalActionMask[i])
            {
                LegalCount++;
            }
        }

        if (LegalCount == 0)
        {
            return -1;
        }

        float InvLegal = 1.0f / (float)LegalCount;

        for (int32 i = 0; i < ActionSize; i++)
        {
            if (Root.LegalActionMask.IsValidIndex(i) && Root.LegalActionMask[i])
            {
                Prob[i] = InvLegal;
            }
        }
    }
    else
    {
        float InvNorm = 1.0f / Norm;

        for (int32 i = 0; i < ActionSize; i++)
        {
            Prob[i] *= InvNorm;
        }
    }

    float R = MCTSRandStream.FRand();
    float Acc = 0.0f;

    for (int32 i = 0; i < ActionSize; i++)
    {
        Acc += Prob[i];

        if (R <= Acc)
        {
            return i;
        }
    }

    return ActionSize - 1;
}

void UAIManager::ResetMCTS()
{
    Tree = FMCTSTree();
    Tree.RootIndex = INDEX_NONE;

    if (InferenceQueue)
    {
        InferenceQueue->Clear();
    }
    // RDGCache is intentionally NOT reset here.
    // Model lifetime is managed exclusively via Initialize and RequestModelSwap.
}

void UAIManager::FinalizeMCTS()
{
    if (!Tree.Nodes.IsValidIndex(Tree.RootIndex))
    {
        bRootFinalized = true;
        return;
    }

    if (bRootFinalized)
    {
        return;
    }

    FMCTSNode& RootNode = Tree.GetNode(Tree.RootIndex);

    const bool bPolicyValid = RootNode.PolicyPrior.Num() > 0;

    if (!bPolicyValid)
    {
        bRootFinalized = true;
        ResetMCTS();
        return;
    }

    const float Temperature = GetMCTSTemperature();

    TArray<float> CanonicalPolicy =
        BuildRootPolicyTarget(Tree.RootIndex, Temperature);

    if (CanonicalPolicy.Num() != RootNode.PolicyPrior.Num() || CanonicalPolicy.Num() == 0)
    {
        bRootFinalized = true;
        ResetMCTS();
        return;
    }

    // ----------------------------------------------------------------
    // WRITE CANONICAL VISIT-COUNT POLICY BACK TO ROOT
    // This is the AlphaZero policy target — the improved policy derived
    // from MCTS visit counts, not the raw network prior. It replaces
    // the network prior on the root node so that when GetActionFromMCTS
    // calls StoreSelfPlaySample, the stored PolicyPrior is the canonical
    // MCTS policy target, not the network's raw output.
    // ----------------------------------------------------------------
    RootNode.PolicyPrior = CanonicalPolicy;
    RootNode.bPolicyInitialized = true;
    RootNode.bIsExpanded = true;

    for (int32 NodeIndex = 0; NodeIndex < Tree.Nodes.Num(); ++NodeIndex)
    {
        FMCTSNode& Node = Tree.Nodes[NodeIndex];

        if (!Node.bIsExpanded)
        {
            continue;
        }

        Node.bPolicyInitialized = true;
    }

    // ----------------------------------------------------------------
    // NOTE: StoreSelfPlaySample is NOT called here.
    // GetActionFromMCTS is the sole sample emission point, called once
    // per action after FinalizeMCTS completes. This prevents duplicate
    // samples from being emitted per action.
    // ----------------------------------------------------------------

    bRootFinalized = true;
}

// ============================================================
// UAIManager — Async MCTS and Training Pipeline
// ============================================================
// GetActionFromMCTS: augmented to run MCTS on a background
// thread. Returns -1 immediately; result delivered via
// OnMCTSComplete delegate on the game thread.
//
// FinalizeMCTSTrainingPipeline: augmented to run training on
// a background thread. Progress delivered via OnTrainingProgress
// delegate (0.0-1.0). Completion via OnTrainingComplete.
//
// CancelMCTS: sets bCancelRequested flag; RunSimulations checks
// this flag at the start of each outer loop iteration and exits
// cleanly, removing virtual losses and backpropagating.
// ============================================================

// ------------------------------------------------------------
// GetActionFromMCTS — async augmentation
// Existing synchronous body moved into background thread.
// Blueprint binds OnMCTSComplete to receive the action.
// ------------------------------------------------------------
int32 UAIManager::GetActionFromMCTS(
    const TArray<float>& GameState,
    int32 PhaseId,
    int32 PlayerId,
    UObject* InGameState,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    if (!InGameState || GameState.Num() == 0)
    {
        UE_LOG(LogTemp, Error,
            TEXT("MCTS -1: InGameState null or GameState empty, Phase=%d (no broadcast)"),
            PhaseId);
        return -1;
    }

    if (bMCTSRunning)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("GetActionFromMCTS: MCTS already running — ignoring call"));
        return -1;
    }

    // Create UObjects here on the game thread (BeginMCTS runs on the
    // background thread, where NewObject is not safe).
    if (!InferenceQueue)
        InferenceQueue = NewObject<UAI_InferenceQueue>(this);
    if (!RDGCache && EnsureModelsExist())
    {
        RDGCache = NewObject<UAI_RDGCache>(this);
        RDGCache->Initialize(UAI_RDGCache::InferenceRuntimeName);
    }

    // Load a newly trained model if one is waiting. Runs here on the game
    // thread, while no search is running.
    if (RDGCache)
        RDGCache->TickModelHotSwap();

    bMCTSRunning = true;
    bCancelRequested = false;

    // Capture inputs by value for the background thread
    TArray<float>               GameStateCopy = GameState;
    TArray<FTerritoryEntityList> EntityListsCopy = InEntityLists;
    int32                        PhaseIdCopy = PhaseId;
    int32                        PlayerIdCopy = PlayerId;
    TWeakObjectPtr<UObject>      GameStateRef_ = InGameState;

    MCTSFuture = Async(EAsyncExecution::Thread, [this,
        GameStateCopy, PhaseIdCopy, PlayerIdCopy,
        GameStateRef_, EntityListsCopy]() mutable
        {
            // ---- Existing synchronous body (unchanged) ----
            int32 Action = -1;

            if (!GameStateRef_.IsValid())
            {
                UE_LOG(LogTemp, Error,
                    TEXT("MCTS -1: GameStateRef invalid, Phase=%d"), PhaseIdCopy);
                bMCTSRunning = false;
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnMCTSComplete.Broadcast(-1);
                    });
                return;
            }

            bRootFinalized = false;
            GameStateRef = GameStateRef_.Get();

            MCTSTiming::Reset();
            UAI_RDGCache::TimingStats = UAI_RDGCache::FInferenceTimingStats();
            const double SearchStart = FPlatformTime::Seconds();

            BeginMCTS(GameStateCopy, PhaseIdCopy, PlayerIdCopy,
                GameStateRef_.Get(), EntityListsCopy);

            if (MCTSTiming::bLogSearchTiming)
            {
                const double SearchMs = (FPlatformTime::Seconds() - SearchStart) * 1000.0;
                const double InferMs = MCTSTiming::InferenceSeconds * 1000.0;
                const int32 RootVisits = Tree.Nodes.IsValidIndex(Tree.RootIndex)
                    ? Tree.GetNode(Tree.RootIndex).VisitCount : 0;
                UE_LOG(LogTemp, Log,
                    TEXT("MCTS search: Phase=%d Player=%d Time=%.1fms RootVisits=%d | Inference: %d calls, %d samples, %.1fms (%.1f%% of search), avg batch %.1f"),
                    PhaseIdCopy, PlayerIdCopy, SearchMs, RootVisits,
                    MCTSTiming::InferenceCalls, MCTSTiming::InferenceSamples, InferMs,
                    SearchMs > 0.0 ? 100.0 * InferMs / SearchMs : 0.0,
                    MCTSTiming::InferenceCalls > 0
                    ? (double)MCTSTiming::InferenceSamples / MCTSTiming::InferenceCalls : 0.0);

                // GPU stage breakdown (all zero when running on CPU)
                const UAI_RDGCache::FInferenceTimingStats& T = UAI_RDGCache::TimingStats;
                UE_LOG(LogTemp, Log,
                    TEXT("  GPU stages: Prep=%.1fms GameThreadWait=%.1fms Upload=%.1fms Execute=%.1fms Readback=%.1fms"),
                    T.PrepSeconds * 1000.0, T.GameThreadWaitSeconds * 1000.0,
                    T.UploadSeconds * 1000.0, T.ExecuteSeconds * 1000.0,
                    T.ReadbackSeconds * 1000.0);
            }

            if (bCancelRequested)
            {
                UE_LOG(LogTemp, Warning, TEXT("GetActionFromMCTS: cancelled"));
                bMCTSRunning = false;
                bCancelRequested = false;
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnMCTSComplete.Broadcast(-1);
                    });
                return;
            }

            const int32 RootIndex = Tree.RootIndex;
            if (RootIndex == INDEX_NONE || !Tree.Nodes.IsValidIndex(RootIndex))
            {
                UE_LOG(LogTemp, Error,
                    TEXT("MCTS -1: no valid root after BeginMCTS, Phase=%d"), PhaseIdCopy);
                ResetMCTS();
                bMCTSRunning = false;
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnMCTSComplete.Broadcast(-1);
                    });
                return;
            }

            const float Temperature = GetMCTSTemperature();
            Action = SelectFinalActionFromVisits(RootIndex, Temperature);

            if (Action < 0)
            {
                const FMCTSNode& R = Tree.GetNode(RootIndex);
                int32 LegalCount = 0;
                for (bool bLegal : R.LegalActionMask) if (bLegal) LegalCount++;
                UE_LOG(LogTemp, Error,
                    TEXT("MCTS -1: no action selected, Phase=%d A=%d Terminal=%d PolicyInit=%d MaskLen=%d LegalActions=%d"),
                    PhaseIdCopy, R.PendingActionContext.PendingActionA, R.bIsTerminal,
                    R.bPolicyInitialized, R.LegalActionMask.Num(), LegalCount);
            }

            // Forced move (only one legal action): nothing to learn, so no
            // training sample is stored for it.
            bool bForcedMove = false;
            if (Tree.Nodes.IsValidIndex(RootIndex))
            {
                int32 LegalCount = 0;
                for (bool bLegal : Tree.GetNode(RootIndex).LegalActionMask)
                    if (bLegal) LegalCount++;
                bForcedMove = (LegalCount <= 1);
            }

            FinalizeMCTS();

            if (Tree.Nodes.IsValidIndex(RootIndex))
            {
                FMCTSNode& Root = Tree.GetNode(RootIndex);
                Root.bPolicyInitialized = true;
            }

            const bool bValidAction = (Action >= 0);
            if (bValidAction && bForcedMove)
            {
                DiscardedForcedMoveCount++;
            }
            if (bValidAction && !bForcedMove)
            {
                FString SkipReason;

                if (!Tree.Nodes.IsValidIndex(RootIndex))
                {
                    SkipReason = TEXT("tree was reset by FinalizeMCTS");
                }
                else
                {
                    const int32 CountBefore =
                        UAI_ReplayBufferManager::Get().GetSampleCount();

                    UAI_ReplayBufferManager::Get().StoreSelfPlaySample(Tree);

                    if (UAI_ReplayBufferManager::Get().GetSampleCount() == CountBefore)
                    {
                        SkipReason = TEXT("rejected by StoreSelfPlaySample");
                    }
                    else
                    {
                        LastSampledActionRootIndex = RootIndex;
                        bHasEmittedRootSampleThisStep = true;
                        CurrentGameSampleCount++;

                        // Flush to disk every 50 samples to keep memory usage low
                        if (UAI_ReplayBufferManager::Get().GetSamplesInMemory() >=
                            ReplayBufferFlushThreshold)
                        {
                            UAI_ReplayBufferManager::Get().FlushPartialToDisk();
                            AsyncTask(ENamedThreads::GameThread, [this]()
                                {
                                    OnSamplesFlushed.Broadcast();
                                });
                        }
                    }
                }

                if (!SkipReason.IsEmpty())
                {
                    DiscardedRejectedCount++;

                    const FString Msg = FString::Printf(
                        TEXT("SAMPLE NOT STORED: Phase=%d Action=%d Reason=%s"),
                        PhaseIdCopy, Action, *SkipReason);
                    UE_LOG(LogTemp, Warning, TEXT("%s"), *Msg);
                    AsyncTask(ENamedThreads::GameThread, [Msg]()
                        {
                            if (GEngine)
                                GEngine->AddOnScreenDebugMessage(-1, 10.0f, FColor::Red, Msg);
                        });
                }
            }

            if (bValidAction && Tree.Nodes.IsValidIndex(RootIndex))
            {
                // Find child matching selected action and set as new root
                int32 NewRootIndex = INDEX_NONE;
                for (int32 ChildIdx : Tree.GetNode(RootIndex).ChildIndices)
                {
                    if (Tree.Nodes.IsValidIndex(ChildIdx) &&
                        Tree.GetNode(ChildIdx).ActionFromParent == Action)
                    {
                        NewRootIndex = ChildIdx;
                        break;
                    }
                }

                if (NewRootIndex != INDEX_NONE)
                {
                    Tree.RootIndex = NewRootIndex;
                    Tree.GetNode(NewRootIndex).ParentIndex = INDEX_NONE;
                    Tree.GetNode(NewRootIndex).ActionFromParent = -1;
                }
                else
                {
                    ResetMCTS();
                }
            }
            else
            {
                ResetMCTS();
            }

            bMCTSRunning = false;

            // Marshal result back to game thread
            AsyncTask(ENamedThreads::GameThread, [this, Action]()
                {
                    OnMCTSComplete.Broadcast(Action);
                });
        });

    return -1;
}


// ------------------------------------------------------------
// CancelMCTS — sets cancellation flag
// BeginMCTS::RunSimulations checks bCancelRequested at the
// start of each outer loop iteration and exits cleanly.
// ------------------------------------------------------------
void UAIManager::CancelMCTS()
{
    if (bMCTSRunning)
    {
        bCancelRequested = true;
        UE_LOG(LogTemp, Warning, TEXT("CancelMCTS: cancellation requested"));
    }
}

bool UAIManager::IsMCTSRunning() const
{
    return bMCTSRunning;
}

bool UAIManager::IsTrainingRunning() const
{
    return bTrainingRunning;
}

// ------------------------------------------------------------
// FinalizeMCTSTrainingPipeline — async augmentation
// Existing body moved into background thread.
// Parses Python stdout for [train] step=X/Y lines and
// broadcasts OnTrainingProgress(X/Y) to Blueprint.
// Broadcasts OnTrainingComplete when done.
// ------------------------------------------------------------
void UAIManager::FinalizeMCTSTrainingPipeline()
{
    if (bTrainingRunning)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("FinalizeMCTSTrainingPipeline: training already running"));
        return;
    }

    bTrainingRunning = true;

    TrainingFuture = Async(EAsyncExecution::Thread, [this]()
        {
            // ---- Flush any remaining in-memory samples ----
            UAI_ReplayBufferManager::Get().FlushPartialToDisk();

            const FString DatasetPath = GetTotalReplayBufferExportPath();
            if (!FPaths::FileExists(DatasetPath))
            {
                ensureMsgf(false, TEXT("Training dataset not found"));
                bTrainingRunning = false;
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnTrainingComplete.Broadcast();
                    });
                return;
            }

            // ---- Launch Python training as subprocess with stdout capture ----
            const FString PythonScript = FPaths::Combine(
                GetPythonAIDir(),
                TEXT("model.py"));
            const FString OutDir = FPaths::Combine(
                GetPythonAIDir());

            // Training batch size: samples per training step. Smaller uses
            // less GPU memory; keep python.exe's shared GPU memory near 0.
            constexpr int32 TrainingBatchSize = 32;

            // -u: unbuffered output, so progress lines arrive immediately.
            const FString Params = FString::Printf(
                TEXT("-u \"%s\" --dataset \"%s\" --out_dir \"%s\" --batch_size %d"),
                *PythonScript, *DatasetPath, *OutDir, TrainingBatchSize);

            void* PipeRead = nullptr;
            void* PipeWrite = nullptr;
            FPlatformProcess::CreatePipe(PipeRead, PipeWrite);

            FProcHandle ProcHandle = FPlatformProcess::CreateProc(
                TEXT("python"),
                *Params,
                false,  // bLaunchDetached
                true,   // bLaunchHidden (output goes to the Unreal log)
                true,   // bLaunchReallyHidden
                nullptr,
                0,
                nullptr,
                PipeWrite,
                nullptr);

            if (!ProcHandle.IsValid())
            {
                ensureMsgf(false, TEXT("Python training pipeline failed to launch"));
                FPlatformProcess::ClosePipe(PipeRead, PipeWrite);
                bTrainingRunning = false;
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnTrainingComplete.Broadcast();
                    });
                return;
            }

            // ---- Handle one complete line of Python output ----
            // Every line goes to the Unreal log as "[Python] ...".
            // "[train] step=X/Y ..." lines are broadcast as progress.
            auto ProcessLine = [this](const FString& Line)
                {
                    if (Line.IsEmpty())
                        return;

                    UE_LOG(LogTemp, Log, TEXT("[Python] %s"), *Line);

                    const int32 StepPos = Line.Find(TEXT("[train] step="));
                    if (StepPos == INDEX_NONE)
                        return;

                    const int32 NumStart = StepPos + 13;   // length of "[train] step="
                    const int32 SlashPos = Line.Find(TEXT("/"), ESearchCase::IgnoreCase,
                        ESearchDir::FromStart, NumStart);
                    if (SlashPos == INDEX_NONE)
                        return;
                    int32 EndPos = Line.Find(TEXT(" "), ESearchCase::IgnoreCase,
                        ESearchDir::FromStart, SlashPos);
                    if (EndPos == INDEX_NONE)
                        EndPos = Line.Len();

                    const int32 Current = FCString::Atoi(*Line.Mid(NumStart, SlashPos - NumStart));
                    const int32 Total = FCString::Atoi(*Line.Mid(SlashPos + 1, EndPos - SlashPos - 1));
                    if (Total <= 0)
                        return;

                    const float Progress =
                        FMath::Clamp((float)Current / (float)Total, 0.0f, 1.0f);

                    FTrainingStats Stats;
                    Stats.SamplesInMemory = UAI_ReplayBufferManager::Get().GetSamplesInMemory();
                    Stats.SamplesOnDisk = UAI_ReplayBufferManager::Get().GetSamplesOnDisk();
                    Stats.CurrentEpisodeId = CurrentEpisodeId;
                    Stats.CurrentGameSampleCount = CurrentGameSampleCount;
                    Stats.CurrentStep = Current;
                    Stats.TotalSteps = Total;

                    AsyncTask(ENamedThreads::GameThread, [this, Progress, Stats]()
                        {
                            OnTrainingProgress.Broadcast(Progress, Stats);
                        });
                };

            // ---- Read output while Python runs; handle complete lines only ----
            FString Pending;
            auto ProcessCompleteLines = [&Pending, &ProcessLine]()
                {
                    int32 LastNewline = INDEX_NONE;
                    if (!Pending.FindLastChar(TEXT('\n'), LastNewline))
                        return;
                    TArray<FString> Lines;
                    Pending.Left(LastNewline).ParseIntoArrayLines(Lines);
                    for (const FString& Line : Lines)
                        ProcessLine(Line.TrimStartAndEnd());
                    Pending = Pending.Mid(LastNewline + 1);
                };

            while (FPlatformProcess::IsProcRunning(ProcHandle))
            {
                const FString NewOutput = FPlatformProcess::ReadPipe(PipeRead);
                if (!NewOutput.IsEmpty())
                {
                    Pending += NewOutput;
                    ProcessCompleteLines();
                }
                FPlatformProcess::Sleep(0.1f);
            }

            // ---- Drain whatever is left after Python exits ----
            Pending += FPlatformProcess::ReadPipe(PipeRead);
            ProcessCompleteLines();
            ProcessLine(Pending.TrimStartAndEnd());

            int32 ReturnCode = 0;
            FPlatformProcess::GetProcReturnCode(ProcHandle, &ReturnCode);
            FPlatformProcess::CloseProc(ProcHandle);
            FPlatformProcess::ClosePipe(PipeRead, PipeWrite);

            if (ReturnCode != 0)
            {
                ensureMsgf(false,
                    TEXT("Python training pipeline failed with return code %d"),
                    ReturnCode);
                bTrainingRunning = false;
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnTrainingComplete.Broadcast();
                    });
                return;
            }

            // ---- Model swap ----
            if (RDGCache)
            {
                RDGCache->RequestModelSwap(GetONNXPath(), GetPTPath());
                UAI_ReplayBufferManager::Get().ResetNewSampleCounter();

                // Delete staging file now that training completed successfully
                const FString SessionName =
                    UAI_ReplayBufferManager::Get().GetCurrentStagingSessionName();
                UAI_ReplayBufferManager::Get().DeleteStagingFile(SessionName);
            }

            // Broadcast 100% progress
            {
                FTrainingStats Stats;
                Stats.SamplesInMemory = UAI_ReplayBufferManager::Get().GetSamplesInMemory();
                Stats.SamplesOnDisk = UAI_ReplayBufferManager::Get().GetSamplesOnDisk();
                Stats.CurrentEpisodeId = CurrentEpisodeId;
                Stats.CurrentGameSampleCount = CurrentGameSampleCount;
                Stats.CurrentStep = 0;
                Stats.TotalSteps = 0;

                AsyncTask(ENamedThreads::GameThread, [this, Stats]()
                    {
                        OnTrainingProgress.Broadcast(1.0f, Stats);
                    });
            }

            bTrainingRunning = false;

            AsyncTask(ENamedThreads::GameThread, [this]()
                {
                    OnTrainingComplete.Broadcast();
                });
        });
}

float UAIManager::GetMCTSTemperature() const
{
    // Number of decisions per game played with exploration (temperature 1.0)
    // before switching to the most-visited action. Tune to taste.
    constexpr int32 TRAINING_EXPLORATION_DECISIONS = 200;

    if (MCTSContext.Mode == EMCTSMode::Evaluation)
        return 0.0f;

    // Training: sample in proportion to visits early in each game,
    // then play the most-visited action.
    return (CurrentGameSampleCount < TRAINING_EXPLORATION_DECISIONS) ? 1.0f : 0.0f;
}

bool UAIManager::IsEvaluationMode() const
{
    return MCTSContext.Mode == EMCTSMode::Evaluation;
}

bool UAIManager::IsTrainingMode() const
{
    return MCTSContext.Mode == EMCTSMode::Training;
}

bool UAIManager::ValidateMCTSRuntimeInvariants(const FMCTSNode& Node)
{
    if (Node.PlayerId < 0)
    {
        return false;
    }
    if (Node.PolicyPrior.Num() < 0)
    {
        return false;
    }
    if (Node.VisitCount < 0)
    {
        return false;
    }

    return true;
}

int32 UAIManager::FindExistingNodeByState(
    const TArray<float>& GameState,
    int32 PhaseId,
    int32 PlayerId) const
{
    const int32 NodeFeatureSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;
    if (GameState.Num() < NodeFeatureSize)
        return INDEX_NONE;

    UE_LOG(LogTemp, Warning, TEXT("FindExistingNodeByState: PhaseId=%d PlayerId=%d TreeNodes=%d"),
        PhaseId, PlayerId, Tree.Nodes.Num());

    // Hash node features only for quick pre-filter
    const uint32 NodeFeatureHash = FCrc::MemCrc32(
        GameState.GetData(),
        NodeFeatureSize * sizeof(float),
        0
    );

    // Hash global features for secondary check
    const int32 GlobalOffset = NodeFeatureSize;
    const int32 GlobalSize = GameState.Num() - NodeFeatureSize;
    const uint32 GlobalFeatureHash = (GlobalSize > 0)
        ? FCrc::MemCrc32(
            GameState.GetData() + GlobalOffset,
            GlobalSize * sizeof(float),
            0)
        : 0;

    for (int32 i = 0; i < Tree.Nodes.Num(); i++)
    {
        const FMCTSNode& Node = Tree.Nodes[i];
        if (Node.PhaseId != PhaseId || Node.PlayerId != PlayerId)
            continue;
        if (Node.NodeFeatures.Num() != NodeFeatureSize)
            continue;

        const uint32 CandidateNodeHash = FCrc::MemCrc32(
            Node.NodeFeatures.GetData(),
            NodeFeatureSize * sizeof(float),
            0
        );

        if (CandidateNodeHash != NodeFeatureHash)
            continue;

        // Secondary check: global features
        if (GlobalSize > 0 && Node.GlobalFeatures.Num() == GlobalSize)
        {
            const uint32 CandidateGlobalHash = FCrc::MemCrc32(
                Node.GlobalFeatures.GetData(),
                GlobalSize * sizeof(float),
                0
            );
            if (CandidateGlobalHash != GlobalFeatureHash)
                continue;
        }

        UE_LOG(LogTemp, Warning, TEXT("FindExistingNodeByState: Found match at NodeIndex=%d"), i);
        return i;
    }

    UE_LOG(LogTemp, Warning, TEXT("FindExistingNodeByState: No match found"));
    return INDEX_NONE;
}

bool UAIManager::ValidateInferenceOutputsNumerical(
    int32 BatchSize,
    const TArray<float>& OutHead2,
    const TArray<float>& OutHead3,
    const TArray<float>& OutHead7,
    const TArray<float>& OutHead10,
    const TArray<float>& OutHead13,
    const TArray<float>& OutHead14,
    const TArray<float>& OutHead20,
    const TArray<float>& OutHead49,
    const TArray<float>& OutHead128,
    const TArray<float>& OutHead202,
    const TArray<float>& OutHead330,
    const TArray<float>& OutHead6581,
    const TArray<float>& OutHead988,
    const TArray<float>& OutValue)
{
    if (BatchSize <= 0) return false;
    auto HasNaN = [](const TArray<float>& Arr) -> bool
        {
            for (float V : Arr)
                if (!FMath::IsFinite(V)) return true;
            return false;
        };
    if (HasNaN(OutHead2) || HasNaN(OutHead3) || HasNaN(OutHead7) || HasNaN(OutHead10) || HasNaN(OutHead13) ||
        HasNaN(OutHead14) || HasNaN(OutHead20) || HasNaN(OutHead49) ||
        HasNaN(OutHead128) || HasNaN(OutHead202) ||
        HasNaN(OutHead330) || HasNaN(OutHead6581) || HasNaN(OutHead988) ||
        HasNaN(OutValue))
    {
        UE_LOG(LogTemp, Error, TEXT("ValidateInferenceOutputsNumerical: NaN/Inf detected"));
        return false;
    }
    if (OutHead2.Num() != BatchSize * PolicyHeadSize::Head2 ||
        OutHead3.Num() != BatchSize * PolicyHeadSize::Head3 ||
        OutHead7.Num() != BatchSize * PolicyHeadSize::Head7 ||
        OutHead10.Num() != BatchSize * PolicyHeadSize::Head10 ||
        OutHead13.Num() != BatchSize * PolicyHeadSize::Head13 ||
        OutHead14.Num() != BatchSize * PolicyHeadSize::Head14 ||
        OutHead20.Num() != BatchSize * PolicyHeadSize::Head20 ||
        OutHead49.Num() != BatchSize * PolicyHeadSize::Head49 ||
        OutHead128.Num() != BatchSize * PolicyHeadSize::Head128 ||
        OutHead202.Num() != BatchSize * PolicyHeadSize::Head202 ||
        OutHead330.Num() != BatchSize * PolicyHeadSize::Head330 ||
        OutHead6581.Num() != BatchSize * PolicyHeadSize::Head6581 ||
        OutHead988.Num() != BatchSize * PolicyHeadSize::Head988 ||
        OutValue.Num() != BatchSize * NUM_PLAYERS)
    {
        UE_LOG(LogTemp, Error, TEXT("ValidateInferenceOutputsNumerical: size mismatch"));
        return false;
    }
    return true;
}

int32 UAIManager::PromoteRootToChild(int32 CurrentRootIndex, int32 Action)
{
    if (!Tree.Nodes.IsValidIndex(CurrentRootIndex))
    {
        return INDEX_NONE;
    }

    FMCTSNode& OldRoot = Tree.GetNode(CurrentRootIndex);

    int32 NewRootIndex = INDEX_NONE;

    for (int32 ChildIndex : OldRoot.ChildIndices)
    {
        if (!Tree.Nodes.IsValidIndex(ChildIndex))
        {
            continue;
        }

        FMCTSNode& ChildNode = Tree.GetNode(ChildIndex);

        if (ChildNode.ActionFromParent == Action)
        {
            NewRootIndex = ChildIndex;
            break;
        }
    }

    if (NewRootIndex == INDEX_NONE || !Tree.Nodes.IsValidIndex(NewRootIndex))
    {
        return INDEX_NONE;
    }

    Tree.RootIndex = NewRootIndex;

    FMCTSNode& NewRoot = Tree.GetNode(NewRootIndex);

    NewRoot.ParentIndex = INDEX_NONE;
    NewRoot.ActionFromParent = -1;

    ensureMsgf(NewRoot.LegalActionMask.Num() > 0, TEXT("Promoted root missing legal mask"));

    ensureMsgf(
        NewRoot.NodeFeatures.Num() > 0,
        TEXT("Promoted root missing node features")
    );

    NewRoot.VisitCount = FMath::Max(NewRoot.VisitCount, 0);
    NewRoot.TotalValue = NewRoot.TotalValue;

    for (int32 ChildIndex : NewRoot.ChildIndices)
    {
        if (!Tree.Nodes.IsValidIndex(ChildIndex))
        {
            continue;
        }

        FMCTSNode& ChildNode = Tree.GetNode(ChildIndex);

        if (ChildNode.ParentIndex != NewRootIndex)
        {
            ChildNode.ParentIndex = NewRootIndex;
        }
    }

    // Propagate new root's PendingActionContext to all children
    if (NewRoot.PendingActionContext.bIsValid)
    {
        for (int32 ChildIndex : NewRoot.ChildIndices)
        {
            if (!Tree.Nodes.IsValidIndex(ChildIndex))
                continue;
            FMCTSNode& ChildNode = Tree.GetNode(ChildIndex);
            ChildNode.PendingActionContext = NewRoot.PendingActionContext;
        }
    }

    return NewRootIndex;
}

int32 UAIManager::GetActionSizeForPhase(int32 PhaseId)
{
    const EPhaseId    TypedPhase = static_cast<EPhaseId>(PhaseId);
    const EPolicyHead Head = GetPolicyHeadForPhase(TypedPhase);

    ensureMsgf(
        Head == EPolicyHead::Head2 ||
        Head == EPolicyHead::Head3 ||
        Head == EPolicyHead::Head7 ||
        Head == EPolicyHead::Head10 ||
        Head == EPolicyHead::Head13 ||
        Head == EPolicyHead::Head14 ||
        Head == EPolicyHead::Head20 ||
        Head == EPolicyHead::Head49 ||
        Head == EPolicyHead::Head128 ||
        Head == EPolicyHead::Head202 ||
        Head == EPolicyHead::Head330 ||
        Head == EPolicyHead::Head6581 ||
        Head == EPolicyHead::Head988,
        TEXT("Invalid PolicyHead mapping for PhaseId: %d"), PhaseId
    );

    const int32 ActionSize = GetPolicySizeForHead(Head);
    ensureMsgf(ActionSize > 0,
        TEXT("Invalid ActionSize for PhaseId: %d"), PhaseId);
    return ActionSize;
}

int32 UAIManager::GetAdaptiveSimulationCount(int32 RootIndex)
{
    if (!Tree.Nodes.IsValidIndex(RootIndex))
    {
        return MCTSContext.SimulationCount;
    }

    const FMCTSNode& Root = Tree.GetNode(RootIndex);

    const int32 ActionSize = Root.EdgeVisitCount.Num();
    if (ActionSize <= 0)
    {
        return MCTSContext.SimulationCount;
    }

    float MeanQ = 0.0f;
    float MeanQ2 = 0.0f;
    int32 CountedActions = 0;

    for (int32 i = 0; i < ActionSize; i++)
    {
        const int32 Visits =
            Root.EdgeVisitCount.IsValidIndex(i) ? Root.EdgeVisitCount[i] : 0;

        if (Visits <= 0)
            continue;

        const float Sum =
            Root.EdgeValueSum.IsValidIndex(i) ? Root.EdgeValueSum[i] : 0.0f;

        const float Q = Sum / (float)Visits;

        MeanQ += Q;
        MeanQ2 += Q * Q;
        CountedActions++;
    }

    if (CountedActions == 0)
    {
        return MCTSContext.SimulationCount;
    }

    MeanQ /= (float)CountedActions;
    MeanQ2 /= (float)CountedActions;

    const float Variance =
        FMath::Max(0.0f, MeanQ2 - (MeanQ * MeanQ));

    // HARD CLAMPED VARIANCE RESPONSE CURVE
    const float ClampedVariance = FMath::Clamp(Variance, 0.0f, 0.25f);

    const float VarianceScale =
        1.0f + (ClampedVariance * 4.0f);

    float Adaptive =
        (float)MCTSContext.SimulationCount * VarianceScale;

    Adaptive = FMath::Clamp(
        Adaptive,
        8.0f,
        (float)MCTSContext.SimulationCount * 2.0f
    );

    return FMath::RoundToInt(Adaptive);
}

void UAIManager::ValidatePhaseRoutingInvariant(int32 NodeIndex)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
        return;

    const FMCTSNode& Node = Tree.GetNode(NodeIndex);
    const EPhaseId    Phase = static_cast<EPhaseId>(Node.PhaseId);
    const EPolicyHead Head = GetPolicyHeadForPhase(Phase);

    ensureMsgf(
        Head == EPolicyHead::Head2 ||
        Head == EPolicyHead::Head3 ||
        Head == EPolicyHead::Head7 ||
        Head == EPolicyHead::Head10 ||
        Head == EPolicyHead::Head13 ||
        Head == EPolicyHead::Head14 ||
        Head == EPolicyHead::Head20 ||
        Head == EPolicyHead::Head49 ||
        Head == EPolicyHead::Head128 ||
        Head == EPolicyHead::Head202 ||
        Head == EPolicyHead::Head330 ||
        Head == EPolicyHead::Head6581 ||
        Head == EPolicyHead::Head988,
        TEXT("Invalid PolicyHead routing for NodeIndex %d PhaseId %d"),
        NodeIndex, Node.PhaseId
    );

    const int32 ExpectedActionSize = GetPolicySizeForHead(Head);
    ensureMsgf(ExpectedActionSize > 0,
        TEXT("Invalid ExpectedActionSize for NodeIndex %d"), NodeIndex);

    if (Node.PolicyPrior.Num() > 0)
    {
        ensureMsgf(
            Node.PolicyPrior.Num() == ExpectedActionSize,
            TEXT("PolicyPrior size mismatch NodeIndex %d"), NodeIndex);
    }

    // Fires only when a node has a legal action mask whose length does not
    // match the action space of its own phase.
    if (Node.LegalActionMask.Num() > 0 &&
        Node.LegalActionMask.Num() != ExpectedActionSize)
    {
        const int32 ParentPhase =
            Tree.Nodes.IsValidIndex(Node.ParentIndex)
            ? Tree.GetNode(Node.ParentIndex).PhaseId
            : INDEX_NONE;

        const FString Msg = FString::Printf(
            TEXT("LegalActionMask size mismatch: NodeIndex=%d Phase=%d MaskLen=%d Expected=%d IsRoot=%d ParentIndex=%d ParentPhase=%d ActionFromParent=%d"),
            NodeIndex, Node.PhaseId, Node.LegalActionMask.Num(), ExpectedActionSize,
            NodeIndex == Tree.RootIndex ? 1 : 0,
            Node.ParentIndex, ParentPhase, Node.ActionFromParent);

        // Logged every time it happens (the ensure below only fires once per session).
        UE_LOG(LogTemp, Error, TEXT("%s"), *Msg);
        ensureMsgf(false, TEXT("%s"), *Msg);
    }
}

bool UAIManager::ExportReplayBuffer()
{
    if (CachedFinalOutcomeValues.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("ExportReplayBuffer called before EndAIArenaGame — no outcome values set. Call EndAIArenaGame first."));
        return false;
    }

    return UAI_ReplayBufferManager::Get().ExportReplayBuffer(CachedFinalOutcomeValues);
}

const FMCTSTree& UAIManager::GetTreeForReplay() const
{
    return Tree;
}

TArray<float> UAIManager::BuildRootPolicyTarget(int32 RootIndex, float Temperature)
{
    TArray<float> OutPolicy;

    if (!Tree.Nodes.IsValidIndex(RootIndex))
    {
        return OutPolicy;
    }

    const FMCTSNode& RootNode = Tree.GetNode(RootIndex);

    const int32 ActionSpaceSize = RootNode.LegalActionMask.Num();

    if (ActionSpaceSize == 0 || RootNode.VisitCount <= 0)
    {
        return OutPolicy;
    }

    OutPolicy.Init(0.0f, ActionSpaceSize);

    float Sum = 0.0f;

    const bool bUseArgmax = (Temperature <= 0.0f);

    const float InvTemp =
        (bUseArgmax)
        ? 0.0f
        : (1.0f / FMath::Max(Temperature, 1e-6f));

    // ------------------------------------------------------------
    // STRICT LEGAL ACTION MASK ENFORCEMENT
    // ------------------------------------------------------------
    for (int32 i = 0; i < ActionSpaceSize; ++i)
    {
        const bool bLegal =
            RootNode.LegalActionMask.IsValidIndex(i) &&
            RootNode.LegalActionMask[i];

        if (!bLegal)
        {
            OutPolicy[i] = 0.0f;
            continue;
        }

        const int32 N =
            RootNode.EdgeVisitCount.IsValidIndex(i)
            ? RootNode.EdgeVisitCount[i]
            : 0;

        float Adjusted = 0.0f;

        if (bUseArgmax)
        {
            Adjusted = (N > 0) ? 1.0f : 0.0f;
        }
        else
        {
            Adjusted = FMath::Pow((float)FMath::Max(N, 0), InvTemp);
        }

        OutPolicy[i] = Adjusted;
        Sum += Adjusted;
    }

    // ------------------------------------------------------------
    // NORMALIZE OVER LEGAL ACTIONS ONLY
    // ------------------------------------------------------------
    if (Sum > 0.0f)
    {
        const float InvSum = 1.0f / Sum;

        for (int32 i = 0; i < ActionSpaceSize; ++i)
        {
            const bool bLegal =
                RootNode.LegalActionMask.IsValidIndex(i) &&
                RootNode.LegalActionMask[i];

            if (!bLegal)
            {
                OutPolicy[i] = 0.0f;
                continue;
            }

            OutPolicy[i] *= InvSum;
        }
    }
    else
    {
        // Fallback: uniform over legal actions only
        int32 LegalCount = 0;

        for (int32 i = 0; i < ActionSpaceSize; ++i)
        {
            if (RootNode.LegalActionMask.IsValidIndex(i) && RootNode.LegalActionMask[i])
            {
                LegalCount++;
            }
        }

        if (LegalCount == 0)
        {
            return OutPolicy;
        }

        const float InvLegal = 1.0f / (float)LegalCount;

        for (int32 i = 0; i < ActionSpaceSize; ++i)
        {
            if (RootNode.LegalActionMask.IsValidIndex(i) && RootNode.LegalActionMask[i])
            {
                OutPolicy[i] = InvLegal;
            }
        }
    }

    return OutPolicy;
}

void UAIManager::BeginAIArenaGame()
{
    UAI_ReplayBufferManager& ReplayMgr = UAI_ReplayBufferManager::Get();
    ReplayMgr.LoadTrainingMetadata(CurrentEpisodeId);
    ReplayMgr.BeginGameSession();
    CurrentGameSampleCount = 0;
    bEpisodeTerminalLocked = false;
    LastSampledActionRootIndex = INDEX_NONE;
    bHasEmittedRootSampleThisStep = false;
}

bool UAIManager::EndAIArenaGame(const TArray<float>& FinalOutcomeValues)
{
    if (FinalOutcomeValues.Num() != NUM_PLAYERS)
    {
        return false;
    }

    CachedFinalOutcomeValues = FinalOutcomeValues;

    UAI_ReplayBufferManager& ReplayMgr = UAI_ReplayBufferManager::Get();

    // Flush any remaining in-memory samples to staging file
    ReplayMgr.FlushPartialToDisk();

    // Apply outcome values to all staged samples
    const bool bStagingApplied =
        ReplayMgr.ApplyOutcomeValuesToEpisode(FinalOutcomeValues);

    if (!bStagingApplied)
    {
        UE_LOG(LogTemp, Error,
            TEXT("EndAIArenaGame: ApplyOutcomeValuesToEpisode failed"));
        return false;
    }

    // Export remaining in-memory samples with final outcome values
    const bool bExportSuccess =
        ReplayMgr.ExportReplayBuffer(FinalOutcomeValues);

    if (!bExportSuccess)
    {
        UE_LOG(LogTemp, Error,
            TEXT("EndAIArenaGame: ExportReplayBuffer failed"));
        return false;
    }

    bEpisodeTerminalLocked = false;
    CurrentGameSampleCount = 0;
    CurrentEpisodeId++;
    ReplayMgr.SaveTrainingMetadata(CurrentEpisodeId);

    return true;
}

TArray<float> UAIManager::ExtractNodeFeatures(const TArray<float>& StateBuffer)
{
    const int32 NodeFeatureSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;
    TArray<float> NodeFeatures;
    if (!ensureMsgf(StateBuffer.Num() >= NodeFeatureSize + GLOBAL_FEATURE_COUNT,
        TEXT("StateBuffer too small for Graph Transformer input")))
    {
        return NodeFeatures;   // empty: the node is rejected before inference
    }

    NodeFeatures.SetNumUninitialized(NodeFeatureSize);
    FMemory::Memcpy(NodeFeatures.GetData(), StateBuffer.GetData(), NodeFeatureSize * sizeof(float));
    return NodeFeatures;
}

TArray<float> UAIManager::ExtractGlobalFeatures(const TArray<float>& StateBuffer)
{
    const int32 NodeFeatureSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;
    TArray<float> GlobalFeatures;
    if (!ensureMsgf(StateBuffer.Num() >= NodeFeatureSize + GLOBAL_FEATURE_COUNT,
        TEXT("StateBuffer too small for Graph Transformer input")))
    {
        return GlobalFeatures;   // empty: the node is rejected before inference
    }

    GlobalFeatures.SetNumUninitialized(GLOBAL_FEATURE_COUNT);
    FMemory::Memcpy(GlobalFeatures.GetData(),
        StateBuffer.GetData() + NodeFeatureSize,
        GLOBAL_FEATURE_COUNT * sizeof(float));
    return GlobalFeatures;
}

bool UAIManager::RunEndToEndInferenceSmokeTest()
{
    bool bAllPassed = true;
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    UE_LOG(LogTemp, Warning, TEXT("  END-TO-END PIPELINE SMOKE TEST"));
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    // ----------------------------------------------------------------
    // STAGE 1: RDG CACHE INITIALIZATION
    // ----------------------------------------------------------------
    if (!RDGCache)
    {
        RDGCache = NewObject<UAI_RDGCache>(this);
        RDGCache->Initialize(UAI_RDGCache::InferenceRuntimeName);
    }
    const bool bCacheValid = RDGCache != nullptr;
    UE_LOG(LogTemp, Warning, TEXT("[1] RDGCache initialized: %s"),
        bCacheValid ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bCacheValid;
    if (!bCacheValid)
    {
        UE_LOG(LogTemp, Warning, TEXT("ABORT: RDGCache failed to initialize."));
        return false;
    }
    // ----------------------------------------------------------------
    // STAGE 2: RAW GPU INFERENCE
    // ----------------------------------------------------------------
    const int32 NodeFeatureSize = NUM_TERRITORIES * NODE_FEATURE_COUNT;
    const int32 GlobalFeatureSize = GLOBAL_FEATURE_COUNT;
    const int32 EntityFlatSize =
        NUM_TERRITORIES * MAX_UNIT_ENTITIES_PER_NODE * UNIT_ENTITY_FEATURE_COUNT;
    const int32 CountFlatSize = NUM_TERRITORIES;
    TArray<float> SyntheticNodeFeatures;
    SyntheticNodeFeatures.SetNum(NodeFeatureSize);
    for (int32 i = 0; i < NodeFeatureSize; i++)
        SyntheticNodeFeatures[i] = FMath::Sin(i * 0.001f);
    TArray<float> SyntheticGlobalFeatures;
    SyntheticGlobalFeatures.SetNum(GlobalFeatureSize);
    for (int32 i = 0; i < GlobalFeatureSize; i++)
        SyntheticGlobalFeatures[i] = FMath::Cos(i * 0.001f);
    TArray<float> SyntheticEntityTensor;
    SyntheticEntityTensor.Init(0.0f, EntityFlatSize);
    TArray<float> SyntheticEntityCounts;
    SyntheticEntityCounts.Init(0.0f, CountFlatSize);
    TArray<TArray<float>> NodeBatch = { SyntheticNodeFeatures };
    TArray<TArray<float>> GlobalBatch = { SyntheticGlobalFeatures };
    TArray<TArray<float>> EntityTensorBatch = { SyntheticEntityTensor };
    TArray<TArray<float>> EntityCountsBatch = { SyntheticEntityCounts };
    TArray<float>         PhaseIdsBatch = { 0.0f };
    TArray<float> OutH2, OutH3, OutH7, OutH10;
    TArray<float> OutH13, OutH14, OutH20, OutH49;
    TArray<float> OutH128, OutH202, OutH330;
    TArray<float> OutH6581, OutH988;
    TArray<float> OutValue;
    const bool bInferenceSuccess = RDGCache->RunInference(
        NodeBatch, GlobalBatch,
        EntityTensorBatch, EntityCountsBatch,
        PhaseIdsBatch,
        OutH2, OutH3, OutH7, OutH10,
        OutH13, OutH14, OutH20, OutH49,
        OutH128, OutH202, OutH330,
        OutH6581, OutH988,
        OutValue);
    const bool bPolicySizesValid =
        OutH2.Num() == PolicyHeadSize::Head2 &&
        OutH3.Num() == PolicyHeadSize::Head3 &&
        OutH7.Num() == PolicyHeadSize::Head7 &&
        OutH10.Num() == PolicyHeadSize::Head10 &&
        OutH13.Num() == PolicyHeadSize::Head13 &&
        OutH14.Num() == PolicyHeadSize::Head14 &&
        OutH20.Num() == PolicyHeadSize::Head20 &&
        OutH49.Num() == PolicyHeadSize::Head49 &&
        OutH128.Num() == PolicyHeadSize::Head128 &&
        OutH202.Num() == PolicyHeadSize::Head202 &&
        OutH330.Num() == PolicyHeadSize::Head330 &&
        OutH6581.Num() == PolicyHeadSize::Head6581 &&
        OutH988.Num() == PolicyHeadSize::Head988;
    const bool bValueSizeValid = OutValue.Num() == NUM_PLAYERS;
    bool bValuesFinite = true;
    for (float V : OutValue)
        if (!FMath::IsFinite(V)) { bValuesFinite = false; break; }
    const bool bInferenceStagePass =
        bInferenceSuccess && bPolicySizesValid && bValueSizeValid && bValuesFinite;
    UE_LOG(LogTemp, Warning, TEXT("[2] GPU Inference:"));
    UE_LOG(LogTemp, Warning, TEXT("    RunInference returned: %s"),
        bInferenceSuccess ? TEXT("TRUE") : TEXT("FALSE"));
    UE_LOG(LogTemp, Warning,
        TEXT("    H2=%d H3=%d H7=%d H10=%d H13=%d H14=%d"),
        OutH2.Num(), OutH3.Num(), OutH7.Num(),
        OutH10.Num(), OutH13.Num(), OutH14.Num());
    UE_LOG(LogTemp, Warning,
        TEXT("    H20=%d H49=%d H128=%d H202=%d H330=%d H6581=%d H988=%d"),
        OutH20.Num(), OutH49.Num(), OutH128.Num(),
        OutH202.Num(), OutH330.Num(),
        OutH6581.Num(), OutH988.Num());
    UE_LOG(LogTemp, Warning, TEXT("    Value size: %d  Finite: %s"),
        OutValue.Num(), bValuesFinite ? TEXT("YES") : TEXT("NO"));
    for (int32 i = 0; i < OutValue.Num(); i++)
        UE_LOG(LogTemp, Warning, TEXT("    Value[%d]: %f"), i, OutValue[i]);
    UE_LOG(LogTemp, Warning, TEXT("[2] Raw Inference: %s"),
        bInferenceStagePass ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bInferenceStagePass;
    // ----------------------------------------------------------------
    // STAGE 3: IN-MEMORY SAMPLE COLLECTION
    // ----------------------------------------------------------------
    UAI_ReplayBufferManager& ReplayMgr = UAI_ReplayBufferManager::Get();
    ReplayMgr.BeginGameSession();
    const int32 SyntheticPhaseId = static_cast<int32>(EPhaseId::PurchaseType);
    const int32 SyntheticPlayerId = 0;
    const int32 SyntheticActionSize = PolicyHeadSize::Head20;
    FMCTSNode SyntheticRoot;
    SyntheticRoot.NodeFeatures = SyntheticNodeFeatures;
    SyntheticRoot.GlobalFeatures = SyntheticGlobalFeatures;
    SyntheticRoot.PhaseId = SyntheticPhaseId;
    SyntheticRoot.PlayerId = SyntheticPlayerId;
    SyntheticRoot.VisitCount = 10;
    SyntheticRoot.bIsExpanded = true;
    SyntheticRoot.bPolicyInitialized = true;
    SyntheticRoot.bIsTerminal = false;
    SyntheticRoot.InitializeEntityLists();
    SyntheticRoot.LegalActionMask.Init(false, SyntheticActionSize);
    for (int32 i = 0; i < FMath::Min(16, SyntheticActionSize); i++)
        SyntheticRoot.LegalActionMask[i] = true;
    SyntheticRoot.EdgeVisitCount.Init(0, SyntheticActionSize);
    SyntheticRoot.VirtualLossEdgeCount.Init(0, SyntheticActionSize);
    for (int32 i = 0; i < FMath::Min(16, SyntheticActionSize); i++)
        SyntheticRoot.EdgeVisitCount[i] = (i % 4) + 1;
    SyntheticRoot.EdgeValueSum.Init(0.0f, SyntheticActionSize);
    SyntheticRoot.PolicyPrior.Init(0.0f, SyntheticActionSize);
    SyntheticRoot.MCTSValuePerPlayer.SetNum(NUM_PLAYERS);
    for (int32 p = 0; p < NUM_PLAYERS; p++)
        SyntheticRoot.MCTSValuePerPlayer[p] =
        OutValue.IsValidIndex(p) ? OutValue[p] : 0.0f;
    SyntheticRoot.ChildIndices.Add(1);
    FMCTSNode SyntheticChild;
    SyntheticChild.ParentIndex = 0;
    SyntheticChild.ActionFromParent = 0;
    FMCTSTree SyntheticTree;
    SyntheticTree.Nodes.Add(SyntheticRoot);
    SyntheticTree.Nodes.Add(SyntheticChild);
    SyntheticTree.RootIndex = 0;
    ReplayMgr.StoreSelfPlaySample(SyntheticTree);
    const int32 SampleCount = ReplayMgr.GetSampleCount();
    const bool  bSampleCollected = SampleCount > 0;
    bool        bSampleValid = false;
    if (bSampleCollected)
    {
        const FMCTSTrainingSample& S = ReplayMgr.GetBuffer()[0];
        bSampleValid = ReplayMgr.IsValidTrainingSample(S);
        UE_LOG(LogTemp, Warning, TEXT("[3] Sample validation:"));
        UE_LOG(LogTemp, Warning, TEXT("    NodeFeatures=%d (expected %d)"),
            S.NodeFeatures.Num(), NodeFeatureSize);
        UE_LOG(LogTemp, Warning, TEXT("    GlobalFeatures=%d (expected %d)"),
            S.GlobalFeatures.Num(), GlobalFeatureSize);
        UE_LOG(LogTemp, Warning, TEXT("    PolicyTarget=%d (expected %d)"),
            S.PolicyTarget.Num(), SyntheticActionSize);
        UE_LOG(LogTemp, Warning, TEXT("    ValueTarget=%d (expected %d)"),
            S.ValueTarget.Num(), NUM_PLAYERS);
        UE_LOG(LogTemp, Warning, TEXT("    PhaseId=%d PlayerId=%d VisitCount=%f"),
            S.PhaseId, S.PlayerId, S.VisitCount);
    }
    UE_LOG(LogTemp, Warning, TEXT("[3] Sample Collection: %s  IsValid: %s"),
        bSampleCollected ? TEXT("PASS") : TEXT("FAIL"),
        bSampleValid ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bSampleCollected && bSampleValid;
    // ----------------------------------------------------------------
    // STAGE 4: REWARD ASSIGNMENT AND REPLAY BUFFER EXPORT
    // ----------------------------------------------------------------
    TArray<float> SyntheticOutcomes;
    SyntheticOutcomes.SetNum(NUM_PLAYERS);
    for (int32 p = 0; p < NUM_PLAYERS; p++)
        SyntheticOutcomes[p] = (p == SyntheticPlayerId) ? 1.0f : -1.0f;
    CachedFinalOutcomeValues = SyntheticOutcomes;
    const bool bExportSuccess = ReplayMgr.ExportReplayBuffer(SyntheticOutcomes);
    const bool bDiskFileExists = FPaths::FileExists(GetTotalReplayBufferExportPath());
    UE_LOG(LogTemp, Warning, TEXT("[4] Replay Buffer Export:"));
    UE_LOG(LogTemp, Warning, TEXT("    ExportReplayBuffer returned: %s"),
        bExportSuccess ? TEXT("TRUE") : TEXT("FALSE"));
    UE_LOG(LogTemp, Warning, TEXT("    Disk file exists: %s  Path: %s"),
        bDiskFileExists ? TEXT("YES") : TEXT("NO"),
        *GetTotalReplayBufferExportPath());
    UE_LOG(LogTemp, Warning, TEXT("[4] Export: %s"),
        (bExportSuccess && bDiskFileExists) ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bExportSuccess && bDiskFileExists;
    // ----------------------------------------------------------------
    // STAGE 5: PYTHON TRAINING PIPELINE
    // ----------------------------------------------------------------
    const FString PythonExe = TEXT("python");
    const FString ModelScript = FPaths::Combine(
        GetPythonAIDir(),
        TEXT("model.py"));
    const FString DatasetPath = GetTotalReplayBufferExportPath();
    const FString OutputDir = FPaths::Combine(
        GetPythonAIDir());
    const FDateTime PTTimeBefore =
        FPaths::FileExists(GetPTPath())
        ? IFileManager::Get().GetTimeStamp(*GetPTPath())
        : FDateTime::MinValue();
    const FDateTime ONNXTimeBefore =
        FPaths::FileExists(GetONNXPath())
        ? IFileManager::Get().GetTimeStamp(*GetONNXPath())
        : FDateTime::MinValue();
    UAI_TrainingPipeline Pipeline;
    const bool bTrainingSuccess = Pipeline.RunPythonTrainingStep(
        PythonExe, ModelScript, DatasetPath, OutputDir);
    const bool     bPTExists = FPaths::FileExists(GetPTPath());
    const bool     bONNXExists = FPaths::FileExists(GetONNXPath());
    const FDateTime PTTimeAfter = bPTExists
        ? IFileManager::Get().GetTimeStamp(*GetPTPath())
        : FDateTime::MinValue();
    const FDateTime ONNXTimeAfter = bONNXExists
        ? IFileManager::Get().GetTimeStamp(*GetONNXPath())
        : FDateTime::MinValue();
    const bool bPTUpdated = PTTimeAfter > PTTimeBefore;
    const bool bONNXUpdated = ONNXTimeAfter > ONNXTimeBefore;
    UE_LOG(LogTemp, Warning, TEXT("[5] Python Training Pipeline:"));
    UE_LOG(LogTemp, Warning, TEXT("    RunPythonTrainingStep returned: %s"),
        bTrainingSuccess ? TEXT("TRUE") : TEXT("FALSE"));
    UE_LOG(LogTemp, Warning, TEXT("    .pt  exists: %s  updated: %s"),
        bPTExists ? TEXT("YES") : TEXT("NO"),
        bPTUpdated ? TEXT("YES") : TEXT("NO"));
    UE_LOG(LogTemp, Warning, TEXT("    .onnx exists: %s  updated: %s"),
        bONNXExists ? TEXT("YES") : TEXT("NO"),
        bONNXUpdated ? TEXT("YES") : TEXT("NO"));
    UE_LOG(LogTemp, Warning, TEXT("[5] Training Pipeline: %s"),
        (bTrainingSuccess && bPTExists && bONNXExists) ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bTrainingSuccess && bPTExists && bONNXExists;
    // ----------------------------------------------------------------
    // STAGE 6: MODEL SWAP REQUEST
    // ----------------------------------------------------------------
    bool bSwapRequested = false;
    if (RDGCache && bONNXExists && bPTExists)
        bSwapRequested = RDGCache->RequestModelSwap(GetONNXPath(), GetPTPath());
    UE_LOG(LogTemp, Warning, TEXT("[6] Model Swap Request: %s"),
        bSwapRequested ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bSwapRequested;
    // ----------------------------------------------------------------
    // STAGE 7: SUMMARY
    // ----------------------------------------------------------------
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    UE_LOG(LogTemp, Warning, TEXT("  SMOKE TEST FINAL RESULT: %s"),
        bAllPassed ? TEXT("PASS") : TEXT("FAIL"));
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    UE_LOG(LogTemp, Warning,
        TEXT("NOTE: GetActionFromMCTS and SimulateTransition"));
    UE_LOG(LogTemp, Warning,
        TEXT("  require Blueprint wiring and must be validated"));
    UE_LOG(LogTemp, Warning,
        TEXT("  in-game via BP_AAGameState or equivalent."));

    // ----------------------------------------------------------------
    // STAGE 8: CLEANUP TEST ARTIFACTS
    // ----------------------------------------------------------------
    IFileManager::Get().Delete(*GetTotalReplayBufferExportPath());
    const FString StagingPath = FPaths::Combine(
        FPaths::ProjectSavedDir(),
        TEXT("AITraining/Staging"),
        TEXT("Episode_0.json"));
    IFileManager::Get().Delete(*StagingPath);
    UE_LOG(LogTemp, Warning, TEXT("[8] Test artifacts cleaned up"));

    return bAllPassed;
}

bool UAIManager::SampleCombatDice(int32 NodeIndex)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
        return false;
    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    if (!Node.GlobalFeatures.IsValidIndex(GLOBAL_BATTLE_TERRITORY_ID))
        return false;
    const int32 ContestedNodeIdx = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_TERRITORY_ID] * 328.0f);
    if (ContestedNodeIdx < 0 || ContestedNodeIdx >= NUM_TERRITORIES)
        return false;
    int32 AtkUnits[NUM_UNIT_TYPES] = {};
    int32 DefUnits[NUM_UNIT_TYPES] = {};
    for (int32 t = 0; t < NUM_UNIT_TYPES; t++)
    {
        AtkUnits[t] = FMath::RoundToInt(
            Node.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + t] * 20.0f);
        DefUnits[t] = FMath::RoundToInt(
            Node.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + t] * 20.0f);
    }
    const int32 AtkPlayerRaw = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_ATTACKER] * 12.0f) - 1;
    const int32 DefPlayerRaw = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_DEFENDER] * 12.0f) - 1;
    const int32 AtkPlayer = FMath::Clamp(AtkPlayerRaw, 0, NUM_PLAYERS - 1);
    const int32 DefPlayer = FMath::Clamp(DefPlayerRaw, 0, NUM_PLAYERS - 1);
    auto GetTech = [&](int32 Player, int32 TechIndex) -> bool
        {
            const int32 Offset =
                GLOBAL_TECH_OFFSET + Player * TECHS_PER_PLAYER + TechIndex;
            return Node.GlobalFeatures.IsValidIndex(Offset) &&
                Node.GlobalFeatures[Offset] > 0.5f;
        };
    const bool bAtkSuperSubs = GetTech(AtkPlayer, GLOBAL_TECH_SUPER_SUBS);
    const bool bAtkJetFighters = GetTech(AtkPlayer, GLOBAL_TECH_JET_FIGHTERS);
    const bool bAtkHeavyBombers = GetTech(AtkPlayer, GLOBAL_TECH_HEAVY_BOMBERS);
    const bool bAtkImprovedMech = GetTech(AtkPlayer, GLOBAL_TECH_IMPROVED_MECH);
    const bool bAtkAdvArtillery = GetTech(AtkPlayer, GLOBAL_TECH_ADV_ARTILLERY);
    const bool bDefRadar = GetTech(DefPlayer, GLOBAL_TECH_RADAR);
    const bool bDefSuperSubs = GetTech(DefPlayer, GLOBAL_TECH_SUPER_SUBS);
    const int32 RoundNumber = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_ROUND_NUMBER] * 20.0f);
    const bool bAtkHasDestroyer = (AtkUnits[UNIT_DESTROYER] > 0);
    const bool bDefHasDestroyer = (DefUnits[UNIT_DESTROYER] > 0);
    int32 AtkInfBonus = 0;
    int32 AtkMechBonus = 0;
    int32 AtkTacBonus = 0;
    {
        const int32 ArtCount = AtkUnits[UNIT_ARTILLERY];
        const int32 MaxArtSupport = bAtkAdvArtillery ? ArtCount * 2 : ArtCount;
        int32 ArtRemaining = MaxArtSupport;
        AtkInfBonus = FMath::Min(ArtRemaining, AtkUnits[UNIT_INFANTRY]);
        ArtRemaining -= AtkInfBonus;
        AtkMechBonus = bAtkImprovedMech
            ? AtkUnits[UNIT_MECH_INF]
            : FMath::Min(ArtRemaining, AtkUnits[UNIT_MECH_INF]);
        const int32 TacSupport = AtkUnits[UNIT_FIGHTER] + AtkUnits[UNIT_TANK];
        AtkTacBonus = FMath::Min(AtkUnits[UNIT_TAC_BOMBER], TacSupport);
    }
    FRandomStream Rng(FMath::Rand());
    int32 AtkHits = 0;
    int32 DefHits = 0;
    // ---- STEP 1: AA GUN FIRE (round 0 only) ----
    if (RoundNumber == 0 && DefUnits[UNIT_AA_GUN] > 0)
    {
        const int32 AirTargetCount =
            AtkUnits[UNIT_FIGHTER] +
            AtkUnits[UNIT_TAC_BOMBER] +
            AtkUnits[UNIT_STRA_BOMBER];
        int32 ParatrooperCount = 0;
        if (Node.EntityLists.IsValidIndex(ContestedNodeIdx))
        {
            for (const FUnitEntity& E : Node.EntityLists[ContestedNodeIdx].Entities)
            {
                if (E.bIsParatrooper) ParatrooperCount += E.Count;
            }
        }
        const int32 TotalAATargets = AirTargetCount + ParatrooperCount;
        const int32 AAHitValue = bDefRadar
            ? AA_GUN_HIT_VALUE_RADAR : AA_GUN_HIT_VALUE_BASE;
        const int32 MaxShots = DefUnits[UNIT_AA_GUN] * AA_GUN_MAX_TARGETS;
        const int32 NumShots = FMath::Min(MaxShots, TotalAATargets);
        for (int32 s = 0; s < NumShots; s++)
        {
            if (Rng.RandRange(1, 6) <= AAHitValue)
                AtkHits++;
        }
    }
    // ---- STEP 2: SUBMARINE SURPRISE STRIKE ----
    const bool bAtkSubsCanSurprise = !bDefHasDestroyer;
    const bool bDefSubsCanSurprise = !bAtkHasDestroyer;
    if (bAtkSubsCanSurprise)
    {
        const int32 SubAttackVal = bAtkSuperSubs ? 3 : SUB_SURPRISE_ATTACK_HIT;
        for (int32 s = 0; s < AtkUnits[UNIT_SUBMARINE]; s++)
            if (Rng.RandRange(1, 6) <= SubAttackVal)
                AtkHits++;
    }
    if (bDefSubsCanSurprise)
    {
        for (int32 s = 0; s < DefUnits[UNIT_SUBMARINE]; s++)
            if (Rng.RandRange(1, 6) <= SUB_SURPRISE_DEFEND_HIT)
                DefHits++;
    }
    // ---- STEP 3: REGULAR ATTACKER FIRE ----
    for (int32 t = 0; t < NUM_UNIT_TYPES; t++)
    {
        if (t == UNIT_TRANSPORT || t == UNIT_AA_GUN) continue;
        if (t == UNIT_SUBMARINE && bAtkSubsCanSurprise) continue;
        int32 HitThreshold = 0;
        switch (t)
        {
        case UNIT_INFANTRY:
            HitThreshold = (AtkInfBonus > 0) ? 2 : UnitAttack::Infantry;
            if (AtkInfBonus > 0) AtkInfBonus--;
            break;
        case UNIT_ARTILLERY:   HitThreshold = UnitAttack::Artillery;  break;
        case UNIT_MECH_INF:
            HitThreshold = (AtkMechBonus > 0) ? 2 : UnitAttack::MechInf;
            if (AtkMechBonus > 0) AtkMechBonus--;
            break;
        case UNIT_TANK:        HitThreshold = UnitAttack::Tank;       break;
        case UNIT_FIGHTER:
            HitThreshold = bAtkJetFighters ? 4 : UnitAttack::Fighter;
            break;
        case UNIT_TAC_BOMBER:
            HitThreshold = (AtkTacBonus > 0) ? 4 : UnitAttack::TacBomber;
            if (AtkTacBonus > 0) AtkTacBonus--;
            break;
        case UNIT_STRA_BOMBER: HitThreshold = UnitAttack::StraBomber; break;
        case UNIT_SUBMARINE:
            HitThreshold = bAtkSuperSubs ? 3 : UnitAttack::Submarine;
            break;
        case UNIT_DESTROYER:   HitThreshold = UnitAttack::Destroyer;  break;
        case UNIT_CRUISER:     HitThreshold = UnitAttack::Cruiser;    break;
        case UNIT_BATTLESHIP:  HitThreshold = UnitAttack::Battleship; break;
        case UNIT_CARRIER:     HitThreshold = 0;                      break;
        default:               HitThreshold = 0;                      break;
        }
        if (HitThreshold <= 0) continue;
        for (int32 u = 0; u < AtkUnits[t]; u++)
        {
            if (t == UNIT_STRA_BOMBER && bAtkHeavyBombers)
            {
                if (FMath::Min(Rng.RandRange(1, 6), Rng.RandRange(1, 6)) <= HitThreshold)
                    AtkHits++;
            }
            else
            {
                if (Rng.RandRange(1, 6) <= HitThreshold)
                    AtkHits++;
            }
        }
    }
    // ---- STEP 4: REGULAR DEFENDER FIRE ----
    for (int32 t = 0; t < NUM_UNIT_TYPES; t++)
    {
        if (t == UNIT_TRANSPORT || t == UNIT_AA_GUN) continue;
        if (t == UNIT_SUBMARINE && bDefSubsCanSurprise) continue;
        int32 HitThreshold = 0;
        switch (t)
        {
        case UNIT_INFANTRY:    HitThreshold = UnitDefense::Infantry;   break;
        case UNIT_ARTILLERY:   HitThreshold = UnitDefense::Artillery;  break;
        case UNIT_MECH_INF:    HitThreshold = UnitDefense::MechInf;    break;
        case UNIT_TANK:        HitThreshold = UnitDefense::Tank;       break;
        case UNIT_FIGHTER:     HitThreshold = UnitDefense::Fighter;    break;
        case UNIT_TAC_BOMBER:  HitThreshold = UnitDefense::TacBomber;  break;
        case UNIT_STRA_BOMBER: HitThreshold = UnitDefense::StraBomber; break;
        case UNIT_SUBMARINE:
            HitThreshold = bDefSuperSubs ? 2 : UnitDefense::Submarine;
            break;
        case UNIT_DESTROYER:   HitThreshold = UnitDefense::Destroyer;  break;
        case UNIT_CRUISER:     HitThreshold = UnitDefense::Cruiser;    break;
        case UNIT_BATTLESHIP:  HitThreshold = UnitDefense::Battleship; break;
        case UNIT_CARRIER:     HitThreshold = UnitDefense::Carrier;    break;
        default:               HitThreshold = 0;                       break;
        }
        if (HitThreshold <= 0) continue;
        for (int32 u = 0; u < DefUnits[t]; u++)
        {
            if (Rng.RandRange(1, 6) <= HitThreshold)
                DefHits++;
        }
    }
    // Write results — encoded as hits / 280.0f
    Node.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] =
        FMath::Clamp((float)AtkHits / 280.0f, 0.0f, 1.0f);
    Node.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] =
        FMath::Clamp((float)DefHits / 280.0f, 0.0f, 1.0f);
    return true;
}

bool UAIManager::SimulateCombatRoundInternal(int32 NodeIndex)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
        return false;
    if (!SampleCombatDice(NodeIndex))
    {
        UE_LOG(LogTemp, Warning,
            TEXT("[CombatTest] SampleCombatDice failed for NodeIndex %d"),
            NodeIndex);
        return false;
    }
    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    const float AtkHitsNorm = Node.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS];
    const float DefHitsNorm = Node.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS];
    int32 AtkHitsRemaining = FMath::RoundToInt(AtkHitsNorm * 280.0f);
    int32 DefHitsRemaining = FMath::RoundToInt(DefHitsNorm * 280.0f);
    const int32 RoundNum = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_ROUND_NUMBER] * 20.0f);
    const bool bAtkHitsValid =
        AtkHitsRemaining >= 0 && AtkHitsNorm >= 0.0f && AtkHitsNorm <= 1.0f;
    const bool bDefHitsValid =
        DefHitsRemaining >= 0 && DefHitsNorm >= 0.0f && DefHitsNorm <= 1.0f;
    if (!bAtkHitsValid || !bDefHitsValid)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("[CombatTest] Round %d: Invalid hit values — "
                "AtkNorm=%.4f DefNorm=%.4f"),
            RoundNum, AtkHitsNorm, DefHitsNorm);
        return false;
    }
    UE_LOG(LogTemp, Log,
        TEXT("[CombatTest] Round %d: AtkHits=%d  DefHits=%d"),
        RoundNum, AtkHitsRemaining, DefHitsRemaining);
    const int32 CasualtyTypeSize = PolicyHeadSize::Head14;
    const int32 CasualtyQuantitySize = PolicyHeadSize::Head20;
    if (Node.PolicyPrior.Num() != CasualtyTypeSize)
        Node.PolicyPrior.Init(1.0f / (float)CasualtyTypeSize, CasualtyTypeSize);
    if (Node.EdgeVisitCount.Num() != CasualtyTypeSize)
        Node.EdgeVisitCount.Init(0, CasualtyTypeSize);
    Node.VirtualLossEdgeCount.Init(0, CasualtyTypeSize);
    if (Node.EdgeValueSum.Num() != CasualtyTypeSize)
        Node.EdgeValueSum.Init(0.0f, CasualtyTypeSize);
    int32 CasualtyIterations = 0;
    const int32 MaxCasualtyIterations = 20;
    while ((AtkHitsRemaining > 0 || DefHitsRemaining > 0) &&
        CasualtyIterations < MaxCasualtyIterations)
    {
        const TArray<bool> PrevTypeMask = Node.LegalActionMask;
        Node.LegalActionMask.Init(true, CasualtyTypeSize);
        const int32 SelectedType = SelectActionPUCT(NodeIndex, 1.5f);
        Node.LegalActionMask = PrevTypeMask;
        if (SelectedType < 0 || SelectedType >= CasualtyTypeSize)
        {
            UE_LOG(LogTemp, Warning,
                TEXT("[CombatTest] Round %d: CasualtyType PUCT invalid %d"),
                RoundNum, SelectedType);
            return false;
        }
        if (Node.EdgeVisitCount.IsValidIndex(SelectedType))
            Node.EdgeVisitCount[SelectedType]++;
        if (Node.EdgeValueSum.IsValidIndex(SelectedType))
            Node.EdgeValueSum[SelectedType] += 0.1f;
        const TArray<float> PrevPrior = Node.PolicyPrior;
        const TArray<int32> PrevVisits = Node.EdgeVisitCount;
        const TArray<float> PrevValSum = Node.EdgeValueSum;
        const TArray<bool>  PrevMask = Node.LegalActionMask;
        Node.PolicyPrior.Init(1.0f / (float)CasualtyQuantitySize, CasualtyQuantitySize);
        Node.EdgeVisitCount.Init(0, CasualtyQuantitySize);
        Node.EdgeValueSum.Init(0.0f, CasualtyQuantitySize);
        Node.LegalActionMask.Init(true, CasualtyQuantitySize);
        const int32 SelectedQty = SelectActionPUCT(NodeIndex, 1.5f);
        Node.PolicyPrior = PrevPrior;
        Node.EdgeVisitCount = PrevVisits;
        Node.EdgeValueSum = PrevValSum;
        Node.LegalActionMask = PrevMask;
        if (SelectedQty < 0 || SelectedQty >= CasualtyQuantitySize)
        {
            UE_LOG(LogTemp, Warning,
                TEXT("[CombatTest] Round %d: CasualtyQuantity PUCT invalid %d"),
                RoundNum, SelectedQty);
            return false;
        }
        const int32 CasualtyQty = SelectedQty + 1;
        if (AtkHitsRemaining > 0)
            AtkHitsRemaining = FMath::Max(0, AtkHitsRemaining - CasualtyQty);
        else
            DefHitsRemaining = FMath::Max(0, DefHitsRemaining - CasualtyQty);
        CasualtyIterations++;
    }
    if (AtkHitsRemaining > 0 || DefHitsRemaining > 0)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("[CombatTest] Round %d: Casualty loop did not converge "
                "(AtkRemaining=%d DefRemaining=%d)"),
            RoundNum, AtkHitsRemaining, DefHitsRemaining);
        return false;
    }
    const int32 ContinueOrRetreatSize = PolicyHeadSize::Head13;
    const TArray<float> PrevPriorCOR = Node.PolicyPrior;
    const TArray<int32> PrevVisitsCOR = Node.EdgeVisitCount;
    const TArray<float> PrevValCOR = Node.EdgeValueSum;
    const TArray<bool>  PrevMaskCOR = Node.LegalActionMask;
    Node.PolicyPrior.Init(
        1.0f / (float)ContinueOrRetreatSize, ContinueOrRetreatSize);
    Node.EdgeVisitCount.Init(0, ContinueOrRetreatSize);
    Node.EdgeValueSum.Init(0.0f, ContinueOrRetreatSize);
    Node.LegalActionMask.Init(true, ContinueOrRetreatSize);
    const int32 SelectedCOR = SelectActionPUCT(NodeIndex, 1.5f);
    Node.PolicyPrior = PrevPriorCOR;
    Node.EdgeVisitCount = PrevVisitsCOR;
    Node.EdgeValueSum = PrevValCOR;
    Node.LegalActionMask = PrevMaskCOR;
    if (SelectedCOR < 0 || SelectedCOR >= ContinueOrRetreatSize)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("[CombatTest] Round %d: ContinueOrRetreat PUCT invalid %d"),
            RoundNum, SelectedCOR);
        return false;
    }
    UE_LOG(LogTemp, Log,
        TEXT("[CombatTest] Round %d: ContinueOrRetreat action=%d"),
        RoundNum, SelectedCOR);
    // Advance round number — encoded as round / 20.0f
    const int32 NextRound = RoundNum + 1;
    Node.GlobalFeatures[GLOBAL_BATTLE_ROUND_NUMBER] =
        FMath::Clamp((float)NextRound / 20.0f, 0.0f, 1.0f);
    Node.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] = 0.0f;
    Node.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] = 0.0f;
    return true;
}

bool UAIManager::RunCombatResolutionTest(int32 MaxRounds)
{
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    UE_LOG(LogTemp, Warning, TEXT("  COMBAT RESOLUTION TEST"));
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    ResetMCTS();
    Tree = FMCTSTree();
    FMCTSNode TestNode;
    TestNode.PhaseId = static_cast<int32>(EPhaseId::CombatResolveCasualtyType);
    TestNode.PlayerId = 0;
    TestNode.bIsTerminal = false;
    TestNode.bIsExpanded = false;
    TestNode.bPolicyInitialized = false;
    TestNode.VisitCount = 1;
    TestNode.NodeFeatures.Init(0.0f, NUM_TERRITORIES * NODE_FEATURE_COUNT);
    TestNode.GlobalFeatures.Init(0.0f, GLOBAL_FEATURE_COUNT);
    TestNode.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);
    TestNode.LegalActionMask.Init(true, PolicyHeadSize::Head14);
    TestNode.PolicyPrior.Init(
        1.0f / (float)PolicyHeadSize::Head14, PolicyHeadSize::Head14);
    TestNode.EdgeVisitCount.Init(0, PolicyHeadSize::Head14);
    TestNode.VirtualLossEdgeCount.Init(0, PolicyHeadSize::Head14);
    TestNode.EdgeValueSum.Init(0.0f, PolicyHeadSize::Head14);
    TestNode.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] = 1.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_TERRITORY_ID] = 0.0f / 328.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATTACKER] = (0 + 1) / 12.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEFENDER] = (1 + 1) / 12.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_INFANTRY] = 3.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_ARTILLERY] = 2.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_MECH_INF] = 2.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_TANK] = 2.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_FIGHTER] = 2.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_TAC_BOMBER] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_STRA_BOMBER] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_SUBMARINE] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_DESTROYER] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_BATTLESHIP] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_INFANTRY] = 4.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_ARTILLERY] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_TANK] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_FIGHTER] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_AA_GUN] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_SUBMARINE] = 2.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_CARRIER] = 1.0f / 20.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ROUND_NUMBER] = 0.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] = 0.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] = 0.0f;
    TestNode.GlobalFeatures[GLOBAL_BATTLE_SEABORNE] = 0.0f;
    const int32 GermanyBase = GLOBAL_TECH_OFFSET + 0 * TECHS_PER_PLAYER;
    TestNode.GlobalFeatures[GermanyBase + GLOBAL_TECH_HEAVY_BOMBERS] = 1.0f;
    TestNode.GlobalFeatures[GermanyBase + GLOBAL_TECH_SUPER_SUBS] = 1.0f;
    const int32 SovietBase = GLOBAL_TECH_OFFSET + 1 * TECHS_PER_PLAYER;
    TestNode.GlobalFeatures[SovietBase + GLOBAL_TECH_RADAR] = 1.0f;
    const int32 TestNodeIndex = Tree.CreateNode(TestNode);
    Tree.RootIndex = TestNodeIndex;
    UE_LOG(LogTemp, Log, TEXT("[CombatTest] Battle state in GlobalFeatures."));
    UE_LOG(LogTemp, Log, TEXT("[CombatTest] Atk: 3 Inf, 2 Art, 2 Mech, 2 Tank, "
        "2 Ftr, 1 Tac, 1 Stra, 1 Sub, 1 DD, 1 BB"));
    UE_LOG(LogTemp, Log, TEXT("[CombatTest] Def: 4 Inf, 1 Art, 1 Tank, 1 Ftr, "
        "1 AA, 2 Sub, 1 CV"));
    UE_LOG(LogTemp, Log, TEXT("[CombatTest] Atk tech: Heavy Bombers, Super Subs"));
    UE_LOG(LogTemp, Log, TEXT("[CombatTest] Def tech: Radar"));
    UE_LOG(LogTemp, Log, TEXT("------------------------------------------------"));
    bool  bAllRoundsPassed = true;
    int32 RoundsCompleted = 0;
    for (int32 Round = 0; Round < MaxRounds; Round++)
    {
        UE_LOG(LogTemp, Log,
            TEXT("[CombatTest] --- Executing round %d ---"), Round);
        if (!SimulateCombatRoundInternal(TestNodeIndex))
        {
            UE_LOG(LogTemp, Warning,
                TEXT("[CombatTest] Round %d FAILED"), Round);
            bAllRoundsPassed = false;
            break;
        }
        RoundsCompleted++;
        const FMCTSNode& VN = Tree.GetNode(TestNodeIndex);
        if (Round >= 1)
        {
            const float AtkP = VN.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS];
            const float DefP = VN.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS];
            if (AtkP > 1e-5f || DefP > 1e-5f)
            {
                UE_LOG(LogTemp, Warning,
                    TEXT("[CombatTest] INVARIANT FAIL: Pending hits not cleared "
                        "after round %d (Atk=%.4f Def=%.4f)"),
                    Round - 1, AtkP, DefP);
                bAllRoundsPassed = false;
                break;
            }
        }
        // Round number must increment — encoded as round / 20.0f
        const int32 ExpectedRound = Round + 1;
        const int32 ActualRound = FMath::RoundToInt(
            VN.GlobalFeatures[GLOBAL_BATTLE_ROUND_NUMBER] * 20.0f);
        if (ActualRound != ExpectedRound)
        {
            UE_LOG(LogTemp, Warning,
                TEXT("[CombatTest] INVARIANT FAIL: Round number expected %d got %d"),
                ExpectedRound, ActualRound);
            bAllRoundsPassed = false;
            break;
        }
    }
    UE_LOG(LogTemp, Warning, TEXT("------------------------------------------------"));
    UE_LOG(LogTemp, Warning,
        TEXT("[CombatTest] Rounds completed: %d / %d"),
        RoundsCompleted, MaxRounds);
    UE_LOG(LogTemp, Warning, TEXT("[CombatTest] Result: %s"),
        bAllRoundsPassed ? TEXT("PASS") : TEXT("FAIL"));
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    ResetMCTS();
    return bAllRoundsPassed;
}

FApplyActionResult UAIManager::StubSimulateTransition(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    int32 Action,
    int32 NodeIndex,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    FApplyActionResult Result;
    const int32 CallDepth = GetNodeDepth(NodeIndex);

    if (CallDepth >= StubMaxChainDepth)
    {
        Result.bIsTerminal = true;
        Result.OutPhaseId = InPhaseId;
        Result.OutPlayerId = InPlayerId;
        const int32 TotalSize =
            NUM_TERRITORIES * NODE_FEATURE_COUNT + GLOBAL_FEATURE_COUNT;
        Result.OutState.Init(0.0f, TotalSize);
        Result.OutLegalActionMask.Init(
            false, GetPolicySizeForPhase(static_cast<EPhaseId>(InPhaseId)));
        Result.OutEntityLists = InEntityLists;
        if (Result.OutEntityLists.Num() != NUM_TERRITORIES)
            Result.OutEntityLists.SetNum(NUM_TERRITORIES);
        return Result;
    }

    const int32 NextPhaseId = (InPhaseId + 1) % PHASE_COUNT;
    const int32 NextPlayerId =
        (NextPhaseId == 0) ? (InPlayerId + 1) % NUM_PLAYERS : InPlayerId;

    Result.OutPhaseId = NextPhaseId;
    Result.OutPlayerId = NextPlayerId;
    Result.bIsTerminal = false;

    // Build a unique synthetic state for this branch
    const int32 TotalSize = NUM_TERRITORIES * NODE_FEATURE_COUNT + GLOBAL_FEATURE_COUNT;
    const int32 GlobalOffset = NUM_TERRITORIES * NODE_FEATURE_COUNT;

    if (InState.Num() == TotalSize)
    {
        Result.OutState = InState;
    }
    else
    {
        Result.OutState.SetNumUninitialized(TotalSize);
        for (int32 i = 0; i < TotalSize; i++)
        {
            Result.OutState[i] = FMath::Abs(
                FMath::Sin((float)(i + Action + CallDepth) * 0.007f));
        }
    }

    // Stamp turn context to make each child state unique
    if (Result.OutState.Num() > GlobalOffset + 5)
    {
        Result.OutState[GlobalOffset + 0] =
            (float)NextPhaseId / (float)(PHASE_COUNT - 1);
        Result.OutState[GlobalOffset + 1] =
            (float)NextPlayerId / (float)(NUM_PLAYERS - 1);
        Result.OutState[GlobalOffset + 2] =
            FMath::Clamp((float)CallDepth / 20.0f, 0.0f, 1.0f);
        const int32 CurrentActionSize =
            GetPolicySizeForPhase(static_cast<EPhaseId>(InPhaseId));
        Result.OutState[GlobalOffset + 3] = (CurrentActionSize > 0)
            ? FMath::Clamp((float)Action / (float)CurrentActionSize, 0.0f, 1.0f)
            : 0.0f;
    }

    // Legal mask — full legal set for small spaces, 32 evenly spaced for large
    const int32 NextActionSize =
        GetPolicySizeForPhase(static_cast<EPhaseId>(NextPhaseId));
    Result.OutLegalActionMask.Init(false, NextActionSize);
    if (NextActionSize > 0)
    {
        const int32 NumLegal = (NextActionSize <= 32)
            ? NextActionSize
            : 32;
        const int32 Stride = FMath::Max(1, NextActionSize / NumLegal);
        for (int32 k = 0; k < NumLegal; k++)
        {
            Result.OutLegalActionMask[(k * Stride) % NextActionSize] = true;
        }
    }

    Result.OutEntityLists = InEntityLists;
    if (Result.OutEntityLists.Num() != NUM_TERRITORIES)
        Result.OutEntityLists.SetNum(NUM_TERRITORIES);

    return Result;
}

FApplyActionResult UAIManager::StubGetCasualtyAssignmentMask(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    FApplyActionResult Result;
    Result.OutPhaseId = InPhaseId;
    Result.OutPlayerId = InPlayerId;
    Result.bIsTerminal = false;

    const int32 ActionSize = PolicyHeadSize::Head14;
    Result.OutLegalActionMask.Init(false, ActionSize);

    // Global feature offset for battle state
    // InState layout: [0..6250] = node features, [6251..6798] = global features
    const int32 GlobalOffset = NUM_TERRITORIES * NODE_FEATURE_COUNT;

    const auto GlobalAt = [&](int32 AbsGlobalIndex) -> float
        {
            const int32 FlatIdx = GlobalOffset + AbsGlobalIndex;
            return InState.IsValidIndex(FlatIdx) ? InState[FlatIdx] : 0.0f;
        };

    const float AtkPending = GlobalAt(GLOBAL_BATTLE_ATK_HITS);
    const float DefPending = GlobalAt(GLOBAL_BATTLE_DEF_HITS);

    const bool bAssigningDefCasualties = (AtkPending > 1e-5f);
    const bool bAssigningAtkCasualties =
        (!bAssigningDefCasualties && DefPending > 1e-5f);

    if (!bAssigningDefCasualties && !bAssigningAtkCasualties)
        return Result;

    // When attacker has hits pending, defender assigns casualties (and vice versa)
    const int32 UnitsGlobalBase = bAssigningDefCasualties
        ? GLOBAL_BATTLE_DEF_UNITS
        : GLOBAL_BATTLE_ATK_UNITS;

    for (int32 t = 0; t < NUM_UNIT_TYPES && t < ActionSize; t++)
    {
        const float UnitCountNorm = GlobalAt(UnitsGlobalBase + t);
        Result.OutLegalActionMask[t] =
            (FMath::RoundToInt(UnitCountNorm * 20.0f) > 0);
    }

    return Result;
}

bool UAIManager::RunFullMCTSSystemTest()
{
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    UE_LOG(LogTemp, Warning, TEXT("  FULL MCTS SYSTEM TEST"));
    UE_LOG(LogTemp, Warning, TEXT("================================================"));

    bool bAllPassed = true;

    // ----------------------------------------------------------------
    // STAGE 1: INFRASTRUCTURE
    // ----------------------------------------------------------------
    if (!EnsureModelsExist())
    {
        UE_LOG(LogTemp, Warning, TEXT("[MCTSTest] ABORT: Model files do not exist."));
        return false;
    }
    if (!InferenceQueue)
        InferenceQueue = NewObject<UAI_InferenceQueue>(this);
    else
        InferenceQueue->Clear();
    if (!RDGCache)
    {
        RDGCache = NewObject<UAI_RDGCache>(this);
        RDGCache->Initialize(UAI_RDGCache::InferenceRuntimeName);
    }
    UE_LOG(LogTemp, Warning, TEXT("[MCTSTest] [1] Infrastructure: PASS"));

    bUseStubTransitionFunctions = true;

    // ----------------------------------------------------------------
    // STAGE 3: SYNTHETIC ROOT STATE
    // 329*19 + 593 = 6,844 floats
    // ----------------------------------------------------------------
    const int32 RootPhaseId = static_cast<int32>(EPhaseId::Tech);
    const int32 RootPlayerId = 0;
    const int32 TotalStateSize =
        NUM_TERRITORIES * NODE_FEATURE_COUNT + GLOBAL_FEATURE_COUNT;

    TArray<float> RootState;
    RootState.SetNumUninitialized(TotalStateSize);
    for (int32 i = 0; i < TotalStateSize; i++)
        RootState[i] = FMath::Abs(FMath::Sin((float)i * 0.003f));

    // Initialize kamikaze_remaining for Japan
    const int32 KamikazeGlobalOffset =
        NUM_TERRITORIES * NODE_FEATURE_COUNT + GLOBAL_KAMIKAZE_REMAINING;
    if (RootState.IsValidIndex(KamikazeGlobalOffset))
        RootState[KamikazeGlobalOffset] = 1.0f;

    // Clear battle state block [558-592]
    for (int32 i = GLOBAL_BATTLE_TERRITORY_ID; i <= GLOBAL_BATTLE_SEABORNE; i++)
    {
        const int32 Offset = NUM_TERRITORIES * NODE_FEATURE_COUNT + i;
        if (RootState.IsValidIndex(Offset))
            RootState[Offset] = 0.0f;
    }

    TArray<FTerritoryEntityList> TestEntityLists;
    TestEntityLists.SetNum(NUM_TERRITORIES);

    // ----------------------------------------------------------------
    // STAGE 4: CREATE ROOT
    // ----------------------------------------------------------------
    ResetMCTS();
    Tree = FMCTSTree();
    UAI_ReplayBufferManager::Get().BeginGameSession();

    const int32 RootIndex =
        CreateRootNode(RootState, RootPhaseId, RootPlayerId, TestEntityLists);
    const bool bRootValid = Tree.Nodes.IsValidIndex(RootIndex);
    const int32 RootActionSize = bRootValid
        ? GetPolicySizeForPhase(
            static_cast<EPhaseId>(Tree.GetNode(RootIndex).PhaseId))
        : 0;

    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [2] Root node created: %s  PhaseId=%d "
            "PlayerId=%d ActionSize=%d"),
        bRootValid ? TEXT("PASS") : TEXT("FAIL"),
        RootPhaseId, RootPlayerId, RootActionSize);
    bAllPassed &= bRootValid;
    if (!bRootValid) { bUseStubTransitionFunctions = false; ResetMCTS(); return false; }

    // ----------------------------------------------------------------
    // STAGE 5: ROOT EVALUATION
    // ----------------------------------------------------------------
    EvaluateNode(RootIndex);
    FlushInferenceBatch();

    FMCTSNode& RootRef = Tree.GetNode(RootIndex);
    const bool bPolicyInit = RootRef.bPolicyInitialized;
    const bool bPriorValid = RootRef.PolicyPrior.Num() == RootActionSize;

    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [3] Root inference: PolicyInit=%s PriorSize=%d (expected %d)"),
        bPolicyInit ? TEXT("PASS") : TEXT("FAIL"),
        RootRef.PolicyPrior.Num(), RootActionSize);
    bAllPassed &= bPolicyInit && bPriorValid;
    if (!bPolicyInit) { bUseStubTransitionFunctions = false; ResetMCTS(); return false; }

    // ----------------------------------------------------------------
    // STAGE 6: DIRICHLET NOISE
    // ----------------------------------------------------------------
    RootRef.VisitCount = 1;
    if (IsTrainingMode()) ApplyRootDirichletNoise(RootRef, 0.25f, 0.3f);

    float PriorSum = 0.0f;
    for (float P : RootRef.PolicyPrior) PriorSum += P;
    const bool bPriorNormalized =
        FMath::IsNearlyEqual(PriorSum, 1.0f, 0.05f) || !IsTrainingMode();
    UE_LOG(LogTemp, Warning, TEXT("[MCTSTest] [4] Dirichlet noise: PriorSum=%.4f %s"),
        PriorSum, bPriorNormalized ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bPriorNormalized;

    // ----------------------------------------------------------------
    // STAGE 7: TWO-PASS MCTS
    // ----------------------------------------------------------------
    const int32 NumSimulations = MCTSContext.SimulationCount;
    int32 TotalExpansions = 0;
    int32 TotalBackprops = 0;

    for (int32 Sim = 0; Sim < NumSimulations; Sim++)
    {
        int32 Current = RootIndex;
        while (true)
        {
            if (!Tree.Nodes.IsValidIndex(Current)) break;
            FMCTSNode& Node = Tree.GetNode(Current);
            ValidateMCTSRuntimeInvariants(Node);
            if (Node.bIsTerminal || !Node.bPolicyInitialized) break;
            const int32 Action = SelectActionPUCT(Current, 1.5f);
            if (Action < 0) break;
            int32 Next = INDEX_NONE;
            for (int32 ChildIdx : Node.ChildIndices)
                if (Tree.GetNode(ChildIdx).ActionFromParent == Action)
                {
                    Next = ChildIdx; break;
                }
            if (Next == INDEX_NONE)
            {
                if (Node.bIsTerminal) break;
                ExpandNode(Current, Action);
                const FMCTSNode& EN = Tree.GetNode(Current);
                const int32 ChildIndex =
                    EN.ChildIndices.Num() > 0 ? EN.ChildIndices.Last() : INDEX_NONE;
                if (ChildIndex == INDEX_NONE) break;
                EvaluateNode(ChildIndex);
                TotalExpansions++;
                break;
            }
            if (!Tree.GetNode(Next).bPolicyInitialized) break;
            Current = Next;
        }
    }

    const int32 BatchedRequests =
        InferenceQueue ? InferenceQueue->GetQueueSize() : 0;
    FlushInferenceBatch();

    const bool bTreeGrew = Tree.Nodes.Num() > 1;
    const bool bBatchWorked = BatchedRequests > 0;
    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [5a] Pass 1 (expansion): NodesInTree=%d "
            "Expansions=%d BatchedGPURequests=%d"),
        Tree.Nodes.Num(), TotalExpansions, BatchedRequests);
    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [5a] Tree grew=%s BatchWorked=%s  %s"),
        bTreeGrew ? TEXT("YES") : TEXT("NO"),
        bBatchWorked ? TEXT("YES") : TEXT("NO"),
        (bTreeGrew && bBatchWorked) ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bTreeGrew && bBatchWorked;

    if (BatchedRequests <= 1)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("[MCTSTest] [5a] NOTE: Only 1 inference request in batch. "
                "Tree may be shallow due to stub terminal conditions."));
    }

    for (int32 Sim = 0; Sim < NumSimulations; Sim++)
    {
        int32 Current = RootIndex;
        while (true)
        {
            if (!Tree.Nodes.IsValidIndex(Current)) break;
            FMCTSNode& Node = Tree.GetNode(Current);
            ValidateMCTSRuntimeInvariants(Node);
            if (Node.bIsTerminal || !Node.bPolicyInitialized) break;
            const int32 Action = SelectActionPUCT(Current, 1.5f);
            if (Action < 0) break;
            int32 Next = INDEX_NONE;
            for (int32 ChildIdx : Node.ChildIndices)
                if (Tree.GetNode(ChildIdx).ActionFromParent == Action)
                {
                    Next = ChildIdx; break;
                }
            if (Next == INDEX_NONE) break;
            Current = Next;
        }
        if (Tree.Nodes.IsValidIndex(Current))
        {
            FMCTSNode& Leaf = Tree.GetNode(Current);
            if (Leaf.bIsTerminal || Leaf.bPolicyInitialized)
            {
                Backpropagate(Current, RootPlayerId);
                TotalBackprops++;
            }
        }
    }

    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [5b] Pass 2 (exploitation): Backprops=%d  %s"),
        TotalBackprops, TotalBackprops > 0 ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= TotalBackprops > 0;

    // ----------------------------------------------------------------
    // STAGE 8: VISIT COUNT INTEGRITY
    // ----------------------------------------------------------------
    {
        const FMCTSNode& RootForVisits = Tree.GetNode(RootIndex);
        const bool bVisitsAccumulated = RootForVisits.VisitCount > 0;
        int32 EdgeSum = 0;
        for (int32 V : RootForVisits.EdgeVisitCount) EdgeSum += V;
        const bool bEdgeVisitsSumValid = EdgeSum > 0;
        UE_LOG(LogTemp, Warning,
            TEXT("[MCTSTest] [6] Visit counts: RootVisits=%d EdgeSum>0=%s  %s"),
            RootForVisits.VisitCount,
            bEdgeVisitsSumValid ? TEXT("YES") : TEXT("NO"),
            (bVisitsAccumulated && bEdgeVisitsSumValid) ? TEXT("PASS") : TEXT("FAIL"));
        bAllPassed &= bVisitsAccumulated && bEdgeVisitsSumValid;
    }

    // ----------------------------------------------------------------
    // STAGE 9: 2-STEP CHAIN CONDITIONING
    // PurchaseType (Head20) → PurchaseQuantity (Head20)
    // ----------------------------------------------------------------
    bool  bFoundConditionedNode = false;
    bool  bConditioningValueValid = false;
    float FoundConditioningValue = -1.0f;
    {
        FMCTSNode PTNode;
        PTNode.PhaseId = static_cast<int32>(EPhaseId::PurchaseType);
        PTNode.PlayerId = 0;
        PTNode.bIsTerminal = PTNode.bIsExpanded = PTNode.bPolicyInitialized = false;
        PTNode.VisitCount = 1;
        const int32 PTSize = PolicyHeadSize::Head20;
        PTNode.NodeFeatures.Init(0.0f, NUM_TERRITORIES * NODE_FEATURE_COUNT);
        PTNode.GlobalFeatures.Init(0.0f, GLOBAL_FEATURE_COUNT);
        PTNode.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] = 1.0f;
        PTNode.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);
        PTNode.LegalActionMask.Init(true, PTSize);
        PTNode.PolicyPrior.Init(1.0f / PTSize, PTSize);
        PTNode.EdgeVisitCount.Init(0, PTSize);
        PTNode.VirtualLossEdgeCount.Init(0, PTSize);
        PTNode.EdgeValueSum.Init(0.0f, PTSize);
        PTNode.EntityLists.SetNum(NUM_TERRITORIES);
        const int32 PTNodeIdx = Tree.CreateNode(PTNode);

        EvaluateNode(PTNodeIdx);
        FlushInferenceBatch();

        const FMCTSNode& PTN = Tree.GetNode(PTNodeIdx);
        int32 DrivingAction = INDEX_NONE;
        for (int32 a = 0; a < PTN.LegalActionMask.Num(); a++)
            if (PTN.LegalActionMask[a]) { DrivingAction = a; break; }

        UE_LOG(LogTemp, Warning,
            TEXT("[MCTSTest] [9-DEBUG] PurchaseTypeNodeIndex=%d "
                "DrivingAction=%d LegalMaskNum=%d bPolicyInitialized=%s"),
            PTNodeIdx, DrivingAction, PTN.LegalActionMask.Num(),
            PTN.bPolicyInitialized ? TEXT("true") : TEXT("false"));

        int32 PQNodeIdx = INDEX_NONE;
        if (DrivingAction != INDEX_NONE)
        {
            ExpandNode(PTNodeIdx, DrivingAction);
            FlushInferenceBatch();
            const FMCTSNode& PTAfter = Tree.GetNode(PTNodeIdx);
            PQNodeIdx = PTAfter.ChildIndices.Num() > 0
                ? PTAfter.ChildIndices.Last() : INDEX_NONE;
        }

        UE_LOG(LogTemp, Warning,
            TEXT("[MCTSTest] [9-DEBUG] PurchaseQuantityNodeIndex=%d"), PQNodeIdx);

        if (PQNodeIdx != INDEX_NONE && Tree.Nodes.IsValidIndex(PQNodeIdx))
        {
            const FMCTSNode& PQN = Tree.GetNode(PQNodeIdx);
            bFoundConditionedNode =
                static_cast<EPhaseId>(PQN.PhaseId) == EPhaseId::PurchaseQuantity;
            if (bFoundConditionedNode && PQN.GlobalFeatures.IsValidIndex(551))
            {
                FoundConditioningValue = PQN.GlobalFeatures[551];
                const float Expected = FMath::Clamp(
                    (float)DrivingAction / (float)PTSize, 0.0f, 1.0f);
                bConditioningValueValid =
                    FMath::IsFinite(FoundConditioningValue) &&
                    FMath::IsNearlyEqual(FoundConditioningValue, Expected, 1e-4f);
            }
        }
    }

    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [9] Pending sub-action conditioning: "
            "FoundChainFinalNode=%s GlobalFeatures[551]=%.4f Valid=%s  %s"),
        bFoundConditionedNode ? TEXT("YES") : TEXT("NO"),
        FoundConditioningValue,
        bConditioningValueValid ? TEXT("YES") : TEXT("NO"),
        (bFoundConditionedNode && bConditioningValueValid) ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bFoundConditionedNode && bConditioningValueValid;

    // ----------------------------------------------------------------
    // STAGE 9b: 3-STEP CHAIN CONDITIONING
    // CombatMoveSource(Head6581) → LoadUnitsCombat(Head49)
    // → CombatMoveUnitDest(Head6581) → CombatMoveQuantity(Head20)
    // ----------------------------------------------------------------
    bool  bFoundThreeStep = false;
    bool  bThreeStepValid = false;
    bool  bThreeStepCorrect = false;
    float ThreeStepValue = -1.0f;
    int32 CapturedSource = INDEX_NONE;
    int32 CapturedUnitDest = INDEX_NONE;
    {
        FMCTSNode CMSNode;
        CMSNode.PhaseId = static_cast<int32>(EPhaseId::CombatMoveSource);
        CMSNode.PlayerId = 0;
        CMSNode.bIsTerminal = CMSNode.bIsExpanded = CMSNode.bPolicyInitialized = false;
        CMSNode.VisitCount = 1;
        const int32 CMSSize = PolicyHeadSize::Head6581;
        CMSNode.NodeFeatures.Init(0.0f, NUM_TERRITORIES * NODE_FEATURE_COUNT);
        CMSNode.GlobalFeatures.Init(0.0f, GLOBAL_FEATURE_COUNT);
        CMSNode.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] = 1.0f;
        CMSNode.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);
        CMSNode.LegalActionMask.Init(true, CMSSize);
        CMSNode.PolicyPrior.Init(1.0f / CMSSize, CMSSize);
        CMSNode.EdgeVisitCount.Init(0, CMSSize);
        CMSNode.VirtualLossEdgeCount.Init(0, CMSSize);
        CMSNode.EdgeValueSum.Init(0.0f, CMSSize);
        CMSNode.EntityLists.SetNum(NUM_TERRITORIES);
        const int32 CMSIdx = Tree.CreateNode(CMSNode);

        CapturedSource = 5;
        ExpandNode(CMSIdx, CapturedSource);
        FlushInferenceBatch();

        const FMCTSNode& CMSAfter = Tree.GetNode(CMSIdx);
        const int32 LUCIdx = CMSAfter.ChildIndices.Num() > 0
            ? CMSAfter.ChildIndices.Last() : INDEX_NONE;

        UE_LOG(LogTemp, Warning,
            TEXT("[MCTSTest] [9b-DEBUG] CMSNodeIndex=%d LUCNodeIndex=%d "
                "LUCPhaseId=%d TreeNodes=%d"),
            CMSIdx, LUCIdx,
            (LUCIdx != INDEX_NONE && Tree.Nodes.IsValidIndex(LUCIdx))
            ? Tree.GetNode(LUCIdx).PhaseId : -1,
            Tree.Nodes.Num());

        int32 CMUDIdx = INDEX_NONE;
        if (LUCIdx != INDEX_NONE && Tree.Nodes.IsValidIndex(LUCIdx))
        {
            EvaluateNode(LUCIdx);
            FlushInferenceBatch();
            CapturedUnitDest = 137;
            ExpandNode(LUCIdx, CapturedUnitDest);
            FlushInferenceBatch();
            const FMCTSNode& LUCAfter = Tree.GetNode(LUCIdx);
            CMUDIdx = LUCAfter.ChildIndices.Num() > 0
                ? LUCAfter.ChildIndices.Last() : INDEX_NONE;
        }

        UE_LOG(LogTemp, Warning,
            TEXT("[MCTSTest] [9b-DEBUG] CMUDNodeIndex=%d PhaseId=%d TreeNodes=%d"),
            CMUDIdx,
            (CMUDIdx != INDEX_NONE && Tree.Nodes.IsValidIndex(CMUDIdx))
            ? Tree.GetNode(CMUDIdx).PhaseId : -1,
            Tree.Nodes.Num());

        int32 CMQIdx = INDEX_NONE;
        if (CMUDIdx != INDEX_NONE && Tree.Nodes.IsValidIndex(CMUDIdx))
        {
            EvaluateNode(CMUDIdx);
            FlushInferenceBatch();
            const FMCTSNode& CMUDNode = Tree.GetNode(CMUDIdx);
            int32 CMUDDriving = 0;
            for (int32 a = 0; a < CMUDNode.LegalActionMask.Num(); a++)
                if (CMUDNode.LegalActionMask[a]) { CMUDDriving = a; break; }

            ExpandNode(CMUDIdx, CMUDDriving);
            FlushInferenceBatch();
            const FMCTSNode& CMUDAfter = Tree.GetNode(CMUDIdx);
            CMQIdx = CMUDAfter.ChildIndices.Num() > 0
                ? CMUDAfter.ChildIndices.Last() : INDEX_NONE;

            UE_LOG(LogTemp, Warning,
                TEXT("[MCTSTest] [9b-DEBUG] CMQNodeIndex=%d PhaseId=%d "
                    "UnitDestAction=%d TreeNodes=%d"),
                CMQIdx,
                (CMQIdx != INDEX_NONE && Tree.Nodes.IsValidIndex(CMQIdx))
                ? Tree.GetNode(CMQIdx).PhaseId : -1,
                CMUDDriving, Tree.Nodes.Num());

            if (CMQIdx != INDEX_NONE && Tree.Nodes.IsValidIndex(CMQIdx))
            {
                const FMCTSNode& CMQNode = Tree.GetNode(CMQIdx);
                bFoundThreeStep =
                    static_cast<EPhaseId>(CMQNode.PhaseId) ==
                    EPhaseId::CombatMoveQuantity;
                if (bFoundThreeStep && CMQNode.GlobalFeatures.IsValidIndex(551))
                {
                    ThreeStepValue = CMQNode.GlobalFeatures[551];
                    bThreeStepValid =
                        FMath::IsFinite(ThreeStepValue) &&
                        ThreeStepValue >= 0.0f && ThreeStepValue <= 1.0f;
                    const float Expected = FMath::Clamp(
                        (float)CMUDDriving / (float)PolicyHeadSize::Head6581,
                        0.0f, 1.0f);
                    bThreeStepCorrect =
                        FMath::IsNearlyEqual(ThreeStepValue, Expected, 1e-4f);
                }
            }
        }
    }

    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [9b] Three-step chain conditioning: "
            "FoundChainFinalNode=%s GlobalFeatures[551]=%.6f "
            "SourceAction=%d UnitDestAction=%d DerivedFromUnitDest=%s  %s"),
        bFoundThreeStep ? TEXT("YES") : TEXT("NO"), ThreeStepValue,
        CapturedSource, CapturedUnitDest,
        bThreeStepCorrect ? TEXT("YES") : TEXT("NO"),
        (bFoundThreeStep && bThreeStepValid && bThreeStepCorrect)
        ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bFoundThreeStep && bThreeStepValid && bThreeStepCorrect;

    // ----------------------------------------------------------------
    // STAGE 9c: TECH DICE SAMPLING
    // ----------------------------------------------------------------
    bool bTechPassed = false;
    for (int32 Trial = 0; Trial < 10 && !bTechPassed; Trial++)
    {
        FMCTSNode TechNode;
        TechNode.PhaseId = static_cast<int32>(EPhaseId::Tech);
        TechNode.PlayerId = 0;
        TechNode.bIsTerminal = TechNode.bIsExpanded = TechNode.bPolicyInitialized = false;
        TechNode.VisitCount = 1;
        TechNode.NodeFeatures.Init(0.0f, NUM_TERRITORIES * NODE_FEATURE_COUNT);
        TechNode.GlobalFeatures.Init(0.0f, GLOBAL_FEATURE_COUNT);
        TechNode.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] = 1.0f;
        TechNode.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);
        TechNode.LegalActionMask.Init(true, PolicyHeadSize::Head7);
        TechNode.PolicyPrior.Init(1.0f / 7.0f, PolicyHeadSize::Head7);
        TechNode.EdgeVisitCount.Init(0, PolicyHeadSize::Head7);
        TechNode.VirtualLossEdgeCount.Init(0, PolicyHeadSize::Head7);
        TechNode.EdgeValueSum.Init(0.0f, PolicyHeadSize::Head7);
        TechNode.EntityLists.SetNum(NUM_TERRITORIES);
        const int32 TechIdx = Tree.CreateNode(TechNode);
        SampleTechDice(TechIdx, 6, 0);
        const FMCTSNode& TR = Tree.GetNode(TechIdx);
        for (int32 t = 0; t < NUM_TECHNOLOGIES; t++)
        {
            const int32 Off = GLOBAL_TECH_OFFSET + 0 * NUM_TECHNOLOGIES + t;
            if (TR.GlobalFeatures.IsValidIndex(Off) && TR.GlobalFeatures[Off] > 0.5f)
            {
                bTechPassed = true; break;
            }
        }
    }
    UE_LOG(LogTemp, Warning, TEXT("[MCTSTest] [9c] Tech dice sampling (6 dice x10): %s"),
        bTechPassed ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bTechPassed;

    // ----------------------------------------------------------------
    // STAGE 9d: KAMIKAZE DICE SAMPLING
    // Battle state written to GlobalFeatures. Verify DEF_HITS in global block.
    // ----------------------------------------------------------------
    bool bKamikazePassed = false;
    for (int32 Trial = 0; Trial < 20 && !bKamikazePassed; Trial++)
    {
        FMCTSNode KamNode;
        KamNode.PhaseId = static_cast<int32>(EPhaseId::KamikazeTarget);
        KamNode.PlayerId = 2;
        KamNode.bIsTerminal = KamNode.bIsExpanded = KamNode.bPolicyInitialized = false;
        KamNode.VisitCount = 1;
        KamNode.NodeFeatures.Init(0.0f, NUM_TERRITORIES * NODE_FEATURE_COUNT);
        KamNode.GlobalFeatures.Init(0.0f, GLOBAL_FEATURE_COUNT);
        KamNode.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] = 1.0f;
        // Set up battle state in global features for SZ6 (territory 207)
        KamNode.GlobalFeatures[GLOBAL_BATTLE_TERRITORY_ID] = 207.0f / 328.0f;
        KamNode.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] = 0.0f;
        KamNode.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);
        KamNode.LegalActionMask.Init(true, PolicyHeadSize::Head20);
        KamNode.PolicyPrior.Init(1.0f / 20.0f, PolicyHeadSize::Head20);
        KamNode.EdgeVisitCount.Init(0, PolicyHeadSize::Head20);
        KamNode.VirtualLossEdgeCount.Init(0, PolicyHeadSize::Head20);
        KamNode.EdgeValueSum.Init(0.0f, PolicyHeadSize::Head20);
        KamNode.EntityLists.SetNum(NUM_TERRITORIES);
        KamNode.PendingActionContext.PendingActionA = 3;
        KamNode.PendingActionContext.PendingActionB = 0;
        KamNode.PendingActionContext.bIsValid = true;
        const int32 KamIdx = Tree.CreateNode(KamNode);
        SampleKamikazeDice(KamIdx);
        const FMCTSNode& KR = Tree.GetNode(KamIdx);
        if (KR.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] > 1e-5f)
            bKamikazePassed = true;
    }
    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [9d] Kamikaze dice sampling (3 strikes, x20): %s"),
        bKamikazePassed ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bKamikazePassed;

    // ----------------------------------------------------------------
    // STAGE 9e: BOMBARDMENT DICE SAMPLING
    // ----------------------------------------------------------------
    bool bBombardPassed = false;
    for (int32 Trial = 0; Trial < 10 && !bBombardPassed; Trial++)
    {
        FMCTSNode BomNode;
        BomNode.PhaseId = static_cast<int32>(EPhaseId::BombardQuantity);
        BomNode.PlayerId = 0;
        BomNode.bIsTerminal = BomNode.bIsExpanded = BomNode.bPolicyInitialized = false;
        BomNode.VisitCount = 1;
        BomNode.NodeFeatures.Init(0.0f, NUM_TERRITORIES * NODE_FEATURE_COUNT);
        BomNode.GlobalFeatures.Init(0.0f, GLOBAL_FEATURE_COUNT);
        BomNode.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] = 1.0f;
        BomNode.GlobalFeatures[GLOBAL_BATTLE_TERRITORY_ID] = 47.0f / 328.0f;
        BomNode.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] = 0.0f;
        BomNode.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);
        BomNode.LegalActionMask.Init(true, PolicyHeadSize::Head20);
        BomNode.PolicyPrior.Init(1.0f / 20.0f, PolicyHeadSize::Head20);
        BomNode.EdgeVisitCount.Init(0, PolicyHeadSize::Head20);
        BomNode.VirtualLossEdgeCount.Init(0, PolicyHeadSize::Head20);
        BomNode.EdgeValueSum.Init(0.0f, PolicyHeadSize::Head20);
        BomNode.EntityLists.SetNum(NUM_TERRITORIES);
        BomNode.PendingActionContext.PendingActionA = 0;
        BomNode.PendingActionContext.PendingActionB = 47;  // 0*201+47 = battleship
        BomNode.PendingActionContext.bIsValid = true;
        BomNode.ActionFromParent = 2;  // 3 ships (0-based)
        const int32 BomIdx = Tree.CreateNode(BomNode);
        SampleBombardmentDice(BomIdx);
        const FMCTSNode& BR = Tree.GetNode(BomIdx);
        if (BR.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] > 1e-5f)
            bBombardPassed = true;
    }
    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [9e] Bombardment dice sampling (3 BBs, x10): %s"),
        bBombardPassed ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bBombardPassed;

    // ----------------------------------------------------------------
    // STAGE 9f: STRATEGIC BOMBING DICE SAMPLING
    // ----------------------------------------------------------------
    bool bSBRPassed = false;
    {
        FMCTSNode SBRNode;
        SBRNode.PhaseId = static_cast<int32>(EPhaseId::StrategicBombingQuantity);
        SBRNode.PlayerId = 0;
        SBRNode.bIsTerminal = SBRNode.bIsExpanded = SBRNode.bPolicyInitialized = false;
        SBRNode.VisitCount = 1;
        SBRNode.NodeFeatures.Init(0.0f, NUM_TERRITORIES * NODE_FEATURE_COUNT);
        SBRNode.GlobalFeatures.Init(0.0f, GLOBAL_FEATURE_COUNT);
        SBRNode.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] = 1.0f;
        SBRNode.GlobalFeatures[GLOBAL_BATTLE_TERRITORY_ID] = 47.0f / 328.0f;
        SBRNode.GlobalFeatures[GLOBAL_BATTLE_ATTACKER] = (0 + 1) / 12.0f;
        SBRNode.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);
        SBRNode.LegalActionMask.Init(true, PolicyHeadSize::Head20);
        SBRNode.PolicyPrior.Init(1.0f / 20.0f, PolicyHeadSize::Head20);
        SBRNode.EdgeVisitCount.Init(0, PolicyHeadSize::Head20);
        SBRNode.VirtualLossEdgeCount.Init(0, PolicyHeadSize::Head20);
        SBRNode.EdgeValueSum.Init(0.0f, PolicyHeadSize::Head20);
        SBRNode.EntityLists.SetNum(NUM_TERRITORIES);
        SBRNode.PendingActionContext.PendingActionA = 2;  // 2 bombers
        SBRNode.PendingActionContext.bIsValid = true;
        SBRNode.ActionFromParent = 1;  // Target IC
        const int32 SBRIdx = Tree.CreateNode(SBRNode);
        SampleStrategicBombingDice(SBRIdx);
        const FMCTSNode& SR = Tree.GetNode(SBRIdx);
        // IC damage at NodeFeatures[47*19+5]
        const int32 ICOffset = 47 * NODE_FEATURE_COUNT + 5;
        bSBRPassed = SR.NodeFeatures.IsValidIndex(ICOffset) &&
            SR.NodeFeatures[ICOffset] > 0.0f;
    }
    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [9f] Strategic bombing dice sampling (2 bombers): %s"),
        bSBRPassed ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bSBRPassed;

    // ----------------------------------------------------------------
    // STAGE 9g: SUBMARINE SURPRISE DICE SAMPLING
    // ----------------------------------------------------------------
    bool bSubPassed = false;
    for (int32 SubTrial = 0; SubTrial < 20 && !bSubPassed; SubTrial++)
    {
        FMCTSNode SubNode;
        SubNode.PhaseId = static_cast<int32>(EPhaseId::SubmarineActionQuantity);
        SubNode.PlayerId = 0;
        SubNode.bIsTerminal = SubNode.bIsExpanded = SubNode.bPolicyInitialized = false;
        SubNode.VisitCount = 1;
        SubNode.NodeFeatures.Init(0.0f, NUM_TERRITORIES * NODE_FEATURE_COUNT);
        SubNode.GlobalFeatures.Init(0.0f, GLOBAL_FEATURE_COUNT);
        SubNode.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] = 1.0f;
        SubNode.GlobalFeatures[GLOBAL_BATTLE_TERRITORY_ID] = 207.0f / 328.0f;
        SubNode.GlobalFeatures[GLOBAL_BATTLE_ATTACKER] = (0 + 1) / 12.0f;
        SubNode.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_SUBMARINE] = 3.0f / 20.0f;
        SubNode.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] = 0.0f;
        SubNode.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] = 0.0f;
        SubNode.MCTSValuePerPlayer.Init(0.0f, NUM_PLAYERS);
        SubNode.LegalActionMask.Init(true, PolicyHeadSize::Head20);
        SubNode.PolicyPrior.Init(1.0f / 20.0f, PolicyHeadSize::Head20);
        SubNode.EdgeVisitCount.Init(0, PolicyHeadSize::Head20);
        SubNode.VirtualLossEdgeCount.Init(0, PolicyHeadSize::Head20);
        SubNode.EdgeValueSum.Init(0.0f, PolicyHeadSize::Head20);
        SubNode.EntityLists.SetNum(NUM_TERRITORIES);
        SubNode.PendingActionContext.PendingActionA = 0;
        SubNode.PendingActionContext.bIsValid = true;
        SubNode.ActionFromParent = 2;
        const int32 SubIdx = Tree.CreateNode(SubNode);
        SampleSubmarineSurpriseDice(SubIdx);
        const FMCTSNode& SubR = Tree.GetNode(SubIdx);
        // Attacker subs fire (no defending destroyer) → ATK_HITS should be set
        bSubPassed = SubR.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] > 1e-5f ||
            SubR.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] > 1e-5f;
    }
    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [9g] Submarine surprise dice sampling (3 subs, SZ6): %s"),
        bSubPassed ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bSubPassed;

    // ----------------------------------------------------------------
    // STAGE 10: FINAL ACTION SELECTION
    // ----------------------------------------------------------------
    const float Temperature = GetMCTSTemperature();

    {
        const FMCTSNode& DiagRoot = Tree.GetNode(RootIndex);
        UE_LOG(LogTemp, Warning,
            TEXT("[MCTSTest] [7-DIAG] RootIndex=%d PhaseId=%d "
                "LegalMaskNum=%d EdgeVisitNum=%d VisitCount=%d"),
            RootIndex, DiagRoot.PhaseId,
            DiagRoot.LegalActionMask.Num(), DiagRoot.EdgeVisitCount.Num(),
            DiagRoot.VisitCount);
        for (int32 a = 0; a < DiagRoot.LegalActionMask.Num(); a++)
        {
            UE_LOG(LogTemp, Warning,
                TEXT("[MCTSTest] [7-DIAG]   action=%d legal=%s visits=%d"),
                a, DiagRoot.LegalActionMask[a] ? TEXT("Y") : TEXT("N"),
                DiagRoot.EdgeVisitCount.IsValidIndex(a)
                ? DiagRoot.EdgeVisitCount[a] : -1);
        }
    }
    UE_LOG(LogTemp, Warning, TEXT("SelectFinalAction: RootIndex=%d EdgeVisitCount[0]=%d EdgeVisitCount[962]=%d"),
        RootIndex,
        Tree.Nodes.IsValidIndex(RootIndex) && Tree.GetNode(RootIndex).EdgeVisitCount.IsValidIndex(0) ? Tree.GetNode(RootIndex).EdgeVisitCount[0] : -1,
        Tree.Nodes.IsValidIndex(RootIndex) && Tree.GetNode(RootIndex).EdgeVisitCount.IsValidIndex(962) ? Tree.GetNode(RootIndex).EdgeVisitCount[962] : -1);
    const int32 FinalAction = SelectFinalActionFromVisits(RootIndex, Temperature);
    const FMCTSNode& FinalRoot = Tree.GetNode(RootIndex);
    const bool bActionValid = (FinalAction >= 0 && FinalAction < RootActionSize);
    const bool bActionLegal = bActionValid &&
        FinalRoot.LegalActionMask.IsValidIndex(FinalAction) &&
        FinalRoot.LegalActionMask[FinalAction];

    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [7] Final action: Action=%d Legal=%s  %s"),
        FinalAction, bActionLegal ? TEXT("YES") : TEXT("NO"),
        (bActionValid && bActionLegal) ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bActionValid && bActionLegal;

    // ----------------------------------------------------------------
    // STAGE 11: TRAINING SAMPLE
    // ----------------------------------------------------------------
    TArray<float> PolicyTarget = BuildRootPolicyTarget(RootIndex, Temperature);
    const bool bPolicyTargetValid = PolicyTarget.Num() == RootActionSize;
    float PolicyTargetSum = 0.0f;
    for (float P : PolicyTarget) PolicyTargetSum += P;

    Tree.GetNode(RootIndex).PolicyPrior = PolicyTarget;
    Tree.GetNode(RootIndex).bPolicyInitialized = true;

    UAI_ReplayBufferManager::Get().StoreSelfPlaySample(Tree);
    const int32 SampleCount = UAI_ReplayBufferManager::Get().GetSampleCount();
    const bool bSampleEmitted = SampleCount > 0;

    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [8] Training sample: PolicyTargetSize=%d "
            "Sum=%.4f SampleCount=%d  %s"),
        PolicyTarget.Num(), PolicyTargetSum, SampleCount,
        (bPolicyTargetValid && bSampleEmitted) ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bPolicyTargetValid && bSampleEmitted;

    // ----------------------------------------------------------------
    // STAGE 12: VALUE HEAD INTEGRITY
    // ----------------------------------------------------------------
    int32 InvalidValueNodes = 0;
    for (int32 i = 0; i < Tree.Nodes.Num(); i++)
    {
        const FMCTSNode& N = Tree.Nodes[i];
        if (!N.bPolicyInitialized) continue;
        if (N.MCTSValuePerPlayer.Num() != NUM_PLAYERS) { InvalidValueNodes++; continue; }
        for (float V : N.MCTSValuePerPlayer)
            if (!FMath::IsFinite(V)) { InvalidValueNodes++; break; }
    }
    UE_LOG(LogTemp, Warning, TEXT("[MCTSTest] [10] Value head integrity: InvalidNodes=%d  %s"),
        InvalidValueNodes, (InvalidValueNodes == 0) ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= (InvalidValueNodes == 0);

    // ----------------------------------------------------------------
    // STAGE 13: TREE REUSE
    // ----------------------------------------------------------------
    const int32 NewRoot = PromoteRootToChild(RootIndex, FinalAction);
    const bool bPromoteValid =
        (NewRoot != INDEX_NONE) &&
        Tree.Nodes.IsValidIndex(NewRoot) &&
        Tree.GetNode(NewRoot).ParentIndex == INDEX_NONE;

    UE_LOG(LogTemp, Warning,
        TEXT("[MCTSTest] [11] Tree reuse (PromoteRootToChild): NewRoot=%d  %s"),
        NewRoot, bPromoteValid ? TEXT("PASS") : TEXT("FAIL"));
    bAllPassed &= bPromoteValid;

    // ----------------------------------------------------------------
    // SUMMARY
    // ----------------------------------------------------------------
    UE_LOG(LogTemp, Warning, TEXT("================================================"));
    UE_LOG(LogTemp, Warning, TEXT("  FULL MCTS SYSTEM TEST RESULT: %s"),
        bAllPassed ? TEXT("PASS") : TEXT("FAIL"));
    UE_LOG(LogTemp, Warning, TEXT("================================================"));

    bUseStubTransitionFunctions = false;
    ResetMCTS();
    return bAllPassed;
}

int32 UAIManager::GetNodeDepth(int32 NodeIndex) const
{
    int32 Depth = 0;
    int32 Current = NodeIndex;

    while (Tree.Nodes.IsValidIndex(Current))
    {
        const FMCTSNode& Node = Tree.Nodes[Current];

        if (Node.ParentIndex == INDEX_NONE)
        {
            break;
        }

        Current = Node.ParentIndex;
        Depth++;

        // Safety bound against malformed cycles — a tree should never
        // be deeper than the total node count.
        if (Depth > Tree.Nodes.Num())
        {
            ensureMsgf(false, TEXT("GetNodeDepth: cycle detected or malformed tree at NodeIndex %d"), NodeIndex);
            break;
        }
    }

    return Depth;
}

FApplyActionResult UAIManager::StubGetLegalActionMask(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    FApplyActionResult Result;
    Result.OutPhaseId = InPhaseId;
    Result.OutPlayerId = InPlayerId;
    Result.bIsTerminal = false;

    const int32 ActionSize =
        GetPolicySizeForPhase(static_cast<EPhaseId>(InPhaseId));
    Result.OutLegalActionMask.Init(false, ActionSize);

    if (ActionSize > 0)
    {
        // Full legal set for small spaces, 32 evenly spaced for large —
        // matches StubSimulateTransition so chain phases have consistent
        // branching and PUCT has real choices to explore.
        const int32 NumLegal = (ActionSize <= 32) ? ActionSize : 32;
        const int32 Stride = FMath::Max(1, ActionSize / NumLegal);
        for (int32 k = 0; k < NumLegal; k++)
        {
            Result.OutLegalActionMask[(k * Stride) % ActionSize] = true;
        }
    }

    return Result;
}

FApplyActionResult UAIManager::Internal_SimulateTransition(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    int32 Action,
    bool bHasPendingContext,
    int32 PendingActionA,
    int32 PendingActionB,
    int32 PendingActionC,
    int32 PendingActionD,
    int32 NodeIndex,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    if (bUseStubTransitionFunctions)
    {
        return StubSimulateTransition(
            InState, InPhaseId, InPlayerId, Action, NodeIndex, InEntityLists);
    }
    return SimulateTransition(
        InState,
        InPhaseId,
        InPlayerId,
        Action,
        bHasPendingContext,
        PendingActionA,
        PendingActionB,
        PendingActionC,
        PendingActionD,
        InEntityLists
    );
}

FApplyActionResult UAIManager::Internal_GetLegalActionMask(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    bool bHasPendingContext,
    int32 PendingActionA,
    int32 PendingActionB,
    int32 PendingActionC,
    int32 PendingActionD,
    int32 NodeIndex,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    if (bUseStubTransitionFunctions)
    {
        return StubGetLegalActionMask(
            InState, InPhaseId, InPlayerId, InEntityLists);
    }
    return GetLegalActionMask(
        InState,
        InPhaseId,
        InPlayerId,
        bHasPendingContext,
        PendingActionA,
        PendingActionB,
        PendingActionC,
        PendingActionD,
        InEntityLists
    );
}

FApplyActionResult UAIManager::Internal_GetCasualtyAssignmentMask(
    const TArray<float>& InState,
    int32 InPhaseId,
    int32 InPlayerId,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    if (bUseStubTransitionFunctions)
    {
        return StubGetCasualtyAssignmentMask(
            InState, InPhaseId, InPlayerId, InEntityLists);
    }
    return GetCasualtyAssignmentMask(InState, InPhaseId, InPlayerId, InEntityLists);
}

bool UAIManager::GetTerritoryInfo(
    int32 NodeIndex,
    FString& OutName,
    TArray<int32>& OutConnectedNodes)
{
    OutName = TEXT("");
    OutConnectedNodes.Empty();

    if (NodeIndex < 0 || NodeIndex >= NUM_TERRITORIES)
    {
        return false;
    }

    // ----------------------------------------------------------------
    // LAZY LOAD AND CACHE
    // Parse AITerritoryGraph.json once; reuse on subsequent calls.
    // ----------------------------------------------------------------
    if (!bTerritoryGraphLoaded)
    {
        bTerritoryGraphLoaded = true;

        const FString GraphPath = FPaths::Combine(
            GetPythonAIDir(),
            TEXT("AITerritoryGraph.json"));

        FString FileData;
        if (!FFileHelper::LoadFileToString(FileData, *GraphPath))
        {
            return false;
        }

        TSharedPtr<FJsonObject> Root;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(FileData);

        if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
        {
            return false;
        }

        // Node names: { "0": "Alaska", ... }
        const TSharedPtr<FJsonObject>* NamesObj = nullptr;
        if (Root->TryGetObjectField(TEXT("node_names"), NamesObj))
        {
            for (const auto& Pair : (*NamesObj)->Values)
            {
                const int32 Idx = FCString::Atoi(*Pair.Key);
                CachedNodeNames.Add(Idx, Pair.Value->AsString());
            }
        }

        // Edges: [[a, b], ...]  — build adjacency lists for both directions
        const TArray<TSharedPtr<FJsonValue>>* EdgesArr = nullptr;
        if (Root->TryGetArrayField(TEXT("edges"), EdgesArr))
        {
            for (const TSharedPtr<FJsonValue>& EdgeVal : *EdgesArr)
            {
                const TArray<TSharedPtr<FJsonValue>>* Pair = nullptr;
                if (!EdgeVal->TryGetArray(Pair) || Pair->Num() != 2)
                    continue;

                const int32 A = (int32)(*Pair)[0]->AsNumber();
                const int32 B = (int32)(*Pair)[1]->AsNumber();

                CachedAdjacency.FindOrAdd(A).AddUnique(B);
                CachedAdjacency.FindOrAdd(B).AddUnique(A);
            }
        }
    }

    const FString* FoundName = CachedNodeNames.Find(NodeIndex);
    if (!FoundName)
    {
        return false;
    }

    OutName = *FoundName;

    if (const TArray<int32>* Neighbors = CachedAdjacency.Find(NodeIndex))
    {
        OutConnectedNodes = *Neighbors;
    }

    return true;
}

bool UAIManager::LoadInitialGameState(
    TArray<float>& OutStateBuffer,
    TArray<FTerritoryEntityList>& OutEntityLists,
    int32& OutPhaseId,
    int32& OutActingPlayer,
    int32& OutRound)
{
    OutStateBuffer.Empty();
    OutEntityLists.Empty();
    OutPhaseId = 0;
    OutActingPlayer = 0;
    OutRound = 1;
    const FString StatePath = FPaths::Combine(
        GetAIDataDir(),
        TEXT("AA1940_Initial_State.json"));
    FString StateData;
    if (!FFileHelper::LoadFileToString(StateData, *StatePath))
    {
        UE_LOG(LogTemp, Error,
            TEXT("LoadInitialGameState: could not read %s"), *StatePath);
        return false;
    }
    TSharedPtr<FJsonObject> StateRoot;
    {
        TSharedRef<TJsonReader<>> Reader =
            TJsonReaderFactory<>::Create(StateData);
        if (!FJsonSerializer::Deserialize(Reader, StateRoot) || !StateRoot.IsValid())
        {
            UE_LOG(LogTemp, Error,
                TEXT("LoadInitialGameState: state JSON parse failed"));
            return false;
        }
    }
    int32 TotalFloats = 0;
    StateRoot->TryGetNumberField(TEXT("total_floats"), TotalFloats);
    // 329*20 + 593 = 7,173
    constexpr int32 ExpectedFloats =
        NUM_TERRITORIES * NODE_FEATURE_COUNT + GLOBAL_FEATURE_COUNT;
    if (TotalFloats != ExpectedFloats)
    {
        UE_LOG(LogTemp, Error,
            TEXT("LoadInitialGameState: total_floats=%d expected=%d"),
            TotalFloats, ExpectedFloats);
        return false;
    }
    StateRoot->TryGetNumberField(TEXT("phase_id"), OutPhaseId);
    StateRoot->TryGetNumberField(TEXT("acting_player"), OutActingPlayer);
    StateRoot->TryGetNumberField(TEXT("round"), OutRound);
    const TArray<TSharedPtr<FJsonValue>>* BufferArr = nullptr;
    if (!StateRoot->TryGetArrayField(TEXT("state_buffer"), BufferArr) || !BufferArr)
    {
        UE_LOG(LogTemp, Error,
            TEXT("LoadInitialGameState: state_buffer array missing"));
        return false;
    }
    OutStateBuffer.Reserve(ExpectedFloats);
    for (const TSharedPtr<FJsonValue>& Val : *BufferArr)
        OutStateBuffer.Add((float)Val->AsNumber());
    if (OutStateBuffer.Num() != ExpectedFloats)
    {
        UE_LOG(LogTemp, Error,
            TEXT("LoadInitialGameState: buffer has %d floats, expected %d"),
            OutStateBuffer.Num(), ExpectedFloats);
        OutStateBuffer.Empty();
        return false;
    }
    // Verify kamikaze_remaining is correctly initialized for Japan.
    // Japan is player 2. Initial value should be 1.0 (6/6).
    const int32 KamikazeOffset =
        NUM_TERRITORIES * NODE_FEATURE_COUNT + GLOBAL_KAMIKAZE_REMAINING;
    if (OutStateBuffer.IsValidIndex(KamikazeOffset))
    {
        if (OutStateBuffer[KamikazeOffset] < 0.99f)
        {
            OutStateBuffer[KamikazeOffset] = 1.0f;
        }
    }
    // ----------------------------------------------------------------
    // Load entity lists
    // ----------------------------------------------------------------
    const FString EntityPath = FPaths::Combine(
        GetAIDataDir(),
        TEXT("AA1940_Initial_Entity_State.json"));
    FString EntityData;
    if (!FFileHelper::LoadFileToString(EntityData, *EntityPath))
    {
        UE_LOG(LogTemp, Error,
            TEXT("LoadInitialGameState: could not read %s"), *EntityPath);
        return false;
    }
    TSharedPtr<FJsonObject> EntityRoot;
    {
        TSharedRef<TJsonReader<>> Reader =
            TJsonReaderFactory<>::Create(EntityData);
        if (!FJsonSerializer::Deserialize(Reader, EntityRoot) || !EntityRoot.IsValid())
        {
            UE_LOG(LogTemp, Error,
                TEXT("LoadInitialGameState: entity JSON parse failed"));
            return false;
        }
    }
    int32 NumTerritories = 0;
    int32 FeatureCount = 0;
    EntityRoot->TryGetNumberField(TEXT("num_territories"), NumTerritories);
    EntityRoot->TryGetNumberField(TEXT("unit_entity_feature_count"), FeatureCount);
    if (NumTerritories != NUM_TERRITORIES)
    {
        UE_LOG(LogTemp, Error,
            TEXT("LoadInitialGameState: entity num_territories=%d expected=%d"),
            NumTerritories, NUM_TERRITORIES);
        return false;
    }
    if (FeatureCount != UNIT_ENTITY_FEATURE_COUNT)
    {
        UE_LOG(LogTemp, Error,
            TEXT("LoadInitialGameState: unit_entity_feature_count=%d expected=%d"),
            FeatureCount, UNIT_ENTITY_FEATURE_COUNT);
        return false;
    }
    OutEntityLists.SetNum(NUM_TERRITORIES);
    const TArray<TSharedPtr<FJsonValue>>* EntityListsArr = nullptr;
    if (!EntityRoot->TryGetArrayField(TEXT("entity_lists"), EntityListsArr) ||
        !EntityListsArr || EntityListsArr->Num() != NUM_TERRITORIES)
    {
        UE_LOG(LogTemp, Error,
            TEXT("LoadInitialGameState: entity_lists missing or wrong size"));
        return false;
    }
    // Helper lambda to safely read a bool field
    auto GetBool = [](const TSharedPtr<FJsonObject>& Obj, const FString& Field) -> bool
        {
            const TSharedPtr<FJsonValue>* Val = Obj->Values.Find(Field);
            return Val && Val->IsValid() && (*Val)->AsBool();
        };
    // Helper lambda to safely read an int32 field
    auto GetInt = [](const TSharedPtr<FJsonObject>& Obj, const FString& Field) -> int32
        {
            double Out = 0.0;
            Obj->TryGetNumberField(Field, Out);
            return FMath::RoundToInt(Out);
        };
    int32 TotalEntitiesLoaded = 0;
    for (int32 t = 0; t < EntityListsArr->Num(); t++)
    {
        const TSharedPtr<FJsonObject> TerritoryObj = (*EntityListsArr)[t]->AsObject();
        if (!TerritoryObj.IsValid()) continue;
        int32 EntityCount = 0;
        TerritoryObj->TryGetNumberField(TEXT("count"), EntityCount);
        if (EntityCount <= 0) continue;
        const TArray<TSharedPtr<FJsonValue>>* EntitiesArr = nullptr;
        if (!TerritoryObj->TryGetArrayField(TEXT("entities"), EntitiesArr) ||
            !EntitiesArr)
            continue;
        FTerritoryEntityList& List = OutEntityLists[t];
        for (const TSharedPtr<FJsonValue>& EntityVal : *EntitiesArr)
        {
            if (List.Entities.Num() >= MAX_UNIT_ENTITIES_PER_NODE)
            {
                UE_LOG(LogTemp, Warning,
                    TEXT("LoadInitialGameState: territory %d exceeds "
                        "MAX_UNIT_ENTITIES_PER_NODE. Truncating."), t);
                break;
            }
            const TSharedPtr<FJsonObject> EntityObj = EntityVal->AsObject();
            if (!EntityObj.IsValid()) continue;
            FUnitEntity Entity;
            Entity.UnitType = GetInt(EntityObj, TEXT("UnitType"));
            Entity.OwningPlayer = GetInt(EntityObj, TEXT("OwningPlayer"));
            Entity.Count = FMath::Max(GetInt(EntityObj, TEXT("Count")), 1);
            Entity.HitPoints = GetInt(EntityObj, TEXT("HitPoints"));
            Entity.MovementRemaining = GetInt(EntityObj, TEXT("MovementRemaining"));
            Entity.SlotId = GetInt(EntityObj, TEXT("SlotId"));
            Entity.bIsScrambled = GetBool(EntityObj, TEXT("bIsScrambled"));
            Entity.bHasLoadedThisTurn = GetBool(EntityObj, TEXT("bHasLoadedThisTurn"));
            Entity.bHasUnloadedThisTurn = GetBool(EntityObj, TEXT("bHasUnloadedThisTurn"));
            Entity.bIsSubmerged = GetBool(EntityObj, TEXT("bIsSubmerged"));
            Entity.CombatEngagementState = GetInt(EntityObj, TEXT("CombatEngagementState"));
            Entity.bIsRetreating = GetBool(EntityObj, TEXT("bIsRetreating"));
            Entity.CargoUnitTypeA = GetInt(EntityObj, TEXT("CargoUnitTypeA"));
            Entity.CargoUnitOwnerA = GetInt(EntityObj, TEXT("CargoUnitOwnerA"));
            Entity.CargoUnitTypeB = GetInt(EntityObj, TEXT("CargoUnitTypeB"));
            Entity.CargoUnitOwnerB = GetInt(EntityObj, TEXT("CargoUnitOwnerB"));
            Entity.bIsStrategicBombing = GetBool(EntityObj, TEXT("bIsStrategicBombing"));
            Entity.bIsEscorting = GetBool(EntityObj, TEXT("bIsEscorting"));
            Entity.bIsIntercepting = GetBool(EntityObj, TEXT("bIsIntercepting"));
            Entity.bIsConductingSurpriseStrike = GetBool(EntityObj, TEXT("bIsConductingSurpriseStrike"));
            Entity.IsBombarding = GetInt(EntityObj, TEXT("IsBombarding"));
            Entity.bHasCompletedSurpriseStrike = GetBool(EntityObj, TEXT("bHasCompletedSurpriseStrike"));
            Entity.bHasCompletedBombardment = GetBool(EntityObj, TEXT("bHasCompletedBombardment"));
            Entity.bIsParatrooper = GetBool(EntityObj, TEXT("bIsParatrooper"));
            Entity.StartOfTurnTerritory = GetInt(EntityObj, TEXT("StartOfTurnTerritory"));
            List.Entities.Add(Entity);
            TotalEntitiesLoaded++;
        }
    }
    return true;
}

void UAIManager::ClearAllEntityListsExcept(int32 ExceptIndex)
{
    for (int32 i = 0; i < Tree.Nodes.Num(); i++)
    {
        if (i == ExceptIndex)
            continue;
        Tree.Nodes[i].bIsExpanded = false;
    }
}

void UAIManager::AssembleEntityTensor(
    const TArray<FTerritoryEntityList>& EntityLists,
    TArray<float>& OutEntityTensor,
    TArray<float>& OutEntityCounts)
{
    const int32 N = NUM_TERRITORIES;
    const int32 E = MAX_UNIT_ENTITIES_PER_NODE;
    const int32 F = UNIT_ENTITY_FEATURE_COUNT;  // 25

    OutEntityTensor.Init(0.0f, N * E * F);
    OutEntityCounts.Init(0.0f, N);

    for (int32 t = 0; t < N; t++)
    {
        if (!EntityLists.IsValidIndex(t)) continue;
        const FTerritoryEntityList& List = EntityLists[t];
        const int32 Count = FMath::Min(List.Entities.Num(), E);
        OutEntityCounts[t] = (float)Count;

        for (int32 e = 0; e < Count; e++)
        {
            float Features[UNIT_ENTITY_FEATURE_COUNT] = {};
            List.Entities[e].ToModelFeatures(Features);
            const int32 Offset = (t * E + e) * F;
            FMemory::Memcpy(
                OutEntityTensor.GetData() + Offset,
                Features,
                F * sizeof(float));
        }
    }
}

bool UAIManager::SampleTechDice(int32 NodeIndex, int32 NumDice, int32 PlayerId)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex) || NumDice <= 0)
        return false;

    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    FRandomStream Rng(FMath::Rand());

    // Tech chart: 6 technologies per chart, two charts (A and B),
    // each die rolls 1-6. A roll of 6 hits the tech chart.
    // On a hit, randomly select one of the 6 techs on a randomly
    // selected chart (A or B). Each chart contains 6 techs.
    // Charts: A = techs 0-5, B = techs 6-11 (per player block).
    // Multiple dice can hit; each hit awards one technology.
    // A tech already researched cannot be re-researched.

    const int32 TechBlockBase = GLOBAL_TECH_OFFSET + PlayerId * NUM_TECHNOLOGIES;

    for (int32 d = 0; d < NumDice; d++)
    {
        const int32 Roll = Rng.RandRange(1, 6);
        if (Roll == 6)
        {
            // Hit — select tech chart (0=A, 1=B) then slot within chart
            const int32 Chart = Rng.RandRange(0, 1);
            const int32 Slot = Rng.RandRange(0, 5);
            const int32 TechIndex = Chart * 6 + Slot;
            const int32 GlobalIdx = TechBlockBase + TechIndex;
            if (Node.GlobalFeatures.IsValidIndex(GlobalIdx))
                Node.GlobalFeatures[GlobalIdx] = 1.0f;
        }
    }
    return true;
}

bool UAIManager::SampleKamikazeDice(int32 NodeIndex)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
        return false;
    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    FRandomStream Rng(FMath::Rand());
    const int32 NumStrikes = Node.PendingActionContext.bIsValid
        ? Node.PendingActionContext.PendingActionA : 0;
    if (NumStrikes <= 0)
        return true;
    const int32 LocationIndex = Node.PendingActionContext.bIsValid
        ? Node.PendingActionContext.PendingActionB : INDEX_NONE;
    if (LocationIndex < 0 || LocationIndex >= 128)
        return false;
    int32 Hits = 0;
    for (int32 s = 0; s < NumStrikes; s++)
    {
        if (Rng.RandRange(1, 6) <= KAMIKAZE_HIT_VALUE)
            Hits++;
    }
    if (Node.GlobalFeatures.IsValidIndex(GLOBAL_BATTLE_DEF_HITS))
    {
        const int32 Existing = FMath::RoundToInt(
            Node.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] * 280.0f);
        Node.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] =
            FMath::Clamp((float)(Existing + Hits) / 280.0f, 0.0f, 1.0f);
    }
    if (Node.GlobalFeatures.IsValidIndex(GLOBAL_KAMIKAZE_REMAINING))
    {
        const int32 Remaining = FMath::RoundToInt(
            Node.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] * (float)KAMIKAZE_TOTAL);
        const int32 NewRemaining = FMath::Max(0, Remaining - NumStrikes);
        Node.GlobalFeatures[GLOBAL_KAMIKAZE_REMAINING] =
            (float)NewRemaining / (float)KAMIKAZE_TOTAL;
    }
    return true;
}

bool UAIManager::SampleBombardmentDice(int32 NodeIndex)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
        return false;
    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    const int32 NumShips = Node.ActionFromParent + 1;
    // PendingActionB = BombardDest encoding (territory 0-201)
    // Ship type is now determined from the entity list via BombardSource (PendingActionA)
    // We default to cruiser hit value; Blueprint sets ship type correctly via entity state
    const int32 HitValue = BOMBARD_CRUISER_HIT;
    FRandomStream Rng(FMath::Rand());
    int32 Hits = 0;
    for (int32 s = 0; s < NumShips; s++)
    {
        if (Rng.RandRange(1, 6) <= HitValue)
            Hits++;
    }
    if (Node.GlobalFeatures.IsValidIndex(GLOBAL_BATTLE_ATK_HITS))
    {
        const int32 Existing = FMath::RoundToInt(
            Node.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] * 280.0f);
        Node.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] =
            FMath::Clamp((float)(Existing + Hits) / 280.0f, 0.0f, 1.0f);
    }
    return true;
}

bool UAIManager::SampleStrategicBombingDice(int32 NodeIndex)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
        return false;
    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    const int32 TerritoryIdx = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_TERRITORY_ID] * 328.0f);
    if (TerritoryIdx < 0 || TerritoryIdx >= NUM_TERRITORIES)
        return false;
    const int32 AtkPlayerRaw = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_ATTACKER] * 12.0f) - 1;
    const int32 AtkPlayer = FMath::Clamp(AtkPlayerRaw, 0, NUM_PLAYERS - 1);
    const int32 DefPlayerRaw = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_DEFENDER] * 12.0f) - 1;
    const int32 DefPlayer = FMath::Clamp(DefPlayerRaw, 0, NUM_PLAYERS - 1);
    auto GetTech = [&](int32 Player, int32 TechIndex) -> bool
        {
            const int32 Offset =
                GLOBAL_TECH_OFFSET + Player * TECHS_PER_PLAYER + TechIndex;
            return Node.GlobalFeatures.IsValidIndex(Offset) &&
                Node.GlobalFeatures[Offset] > 0.5f;
        };
    const bool bHeavyBombers = GetTech(AtkPlayer, GLOBAL_TECH_HEAVY_BOMBERS);
    const bool bDefRadar = GetTech(DefPlayer, GLOBAL_TECH_RADAR);
    const int32 FacilityType = Node.ActionFromParent;
    const int32 NumBombers = Node.PendingActionContext.bIsValid
        ? Node.PendingActionContext.PendingActionA : 1;
    const int32 AAHitValue = bDefRadar ? FACILITY_AA_HIT_RADAR : FACILITY_AA_HIT_BASE;
    FRandomStream Rng(FMath::Rand());
    int32 SurvivingBombers = NumBombers;
    for (int32 b = 0; b < NumBombers; b++)
    {
        if (Rng.RandRange(1, 6) <= AAHitValue)
            SurvivingBombers--;
    }
    if (SurvivingBombers <= 0)
        return true;
    int32 TotalDamage = 0;
    for (int32 b = 0; b < SurvivingBombers; b++)
    {
        int32 Roll = Rng.RandRange(1, 6);
        bool bIsStraBomber = false;
        if (Node.EntityLists.IsValidIndex(TerritoryIdx))
        {
            for (const FUnitEntity& E : Node.EntityLists[TerritoryIdx].Entities)
            {
                if (E.bIsStrategicBombing && E.UnitType == UNIT_STRA_BOMBER)
                {
                    bIsStraBomber = true;
                    break;
                }
            }
        }
        if (bIsStraBomber)
            Roll += STRA_BOMBER_DAMAGE_BONUS;
        if (bHeavyBombers)
            Roll = FMath::Max(Roll, Rng.RandRange(1, 6) + (bIsStraBomber ? STRA_BOMBER_DAMAGE_BONUS : 0));
        TotalDamage += Roll;
    }
    const int32 NodeBase = TerritoryIdx * NODE_FEATURE_COUNT;
    if (!Node.NodeFeatures.IsValidIndex(NodeBase + 5))
        return true;
    bool bIsMajorIC = Node.NodeFeatures.IsValidIndex(NodeBase + 4) &&
        Node.NodeFeatures[NodeBase + 4] > 0.5f;
    int32 DamageCap = BASE_DAMAGE_CAP;
    if (FacilityType == 1)
        DamageCap = bIsMajorIC ? MAJOR_IC_DAMAGE_CAP : MINOR_IC_DAMAGE_CAP;
    const float DamageScale = (FacilityType == 1) ? 20.0f : 6.0f;
    const int32 CurrentDamage = FMath::RoundToInt(
        Node.NodeFeatures[NodeBase + 5] * DamageScale);
    const int32 NewDamage = FMath::Min(CurrentDamage + TotalDamage, DamageCap);
    Node.NodeFeatures[NodeBase + 5] =
        FMath::Clamp((float)NewDamage / DamageScale, 0.0f, 1.0f);
    return true;
}

bool UAIManager::SampleSubmarineSurpriseDice(int32 NodeIndex)
{
    if (!Tree.Nodes.IsValidIndex(NodeIndex))
        return false;
    FMCTSNode& Node = Tree.GetNode(NodeIndex);
    const int32 AtkSubs = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_SUBMARINE] * 20.0f);
    const int32 DefSubs = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_SUBMARINE] * 20.0f);
    const int32 AtkDDs = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_ATK_UNITS + UNIT_DESTROYER] * 20.0f);
    const int32 DefDDs = FMath::RoundToInt(
        Node.GlobalFeatures[GLOBAL_BATTLE_DEF_UNITS + UNIT_DESTROYER] * 20.0f);
    int32 AtkHits = 0;
    int32 DefHits = 0;
    if (AtkSubs > 0 && DefDDs == 0)
    {
        for (int32 i = 0; i < AtkSubs; i++)
            if (FMath::RandRange(1, 6) <= 2)
                AtkHits++;
    }
    if (DefSubs > 0 && AtkDDs == 0)
    {
        for (int32 i = 0; i < DefSubs; i++)
            if (FMath::RandRange(1, 6) <= 1)
                DefHits++;
    }
    // Write results — encoded as hits / 280.0f
    Node.GlobalFeatures[GLOBAL_BATTLE_ATK_HITS] =
        FMath::Clamp((float)AtkHits / 280.0f, 0.0f, 1.0f);
    Node.GlobalFeatures[GLOBAL_BATTLE_DEF_HITS] =
        FMath::Clamp((float)DefHits / 280.0f, 0.0f, 1.0f);
    return (AtkHits > 0 || DefHits > 0);
}

int32 UAIManager::GetActionFromAIModel(
    const TArray<float>& GameState,
    int32 PhaseId,
    int32 PlayerId,
    const TArray<FTerritoryEntityList>& InEntityLists,
    int32 PendingActionA,
    int32 PendingActionB,
    int32 PendingActionC,
    int32 PendingActionD)
{
    if (GameState.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActionFromAIModel: GameState is empty"));
        return -1;
    }

    // ---- Model setup on the game thread (UObjects must not be created
    //      on the background thread) ----
    if (!RDGCache)
    {
        RDGCache = NewObject<UAI_RDGCache>(this);
        if (!RDGCache->Initialize(UAI_RDGCache::InferenceRuntimeName))
        {
            UE_LOG(LogTemp, Error,
                TEXT("GetActionFromAIModel: RDGCache failed to initialize"));
            RDGCache = nullptr;
            return -1;
        }
    }
    else
    {
        // Load a newly trained model if one is waiting
        RDGCache->TickModelHotSwap();
    }

    // Capture inputs by value for the background thread
    TArray<float>                GameStateCopy = GameState;
    TArray<FTerritoryEntityList> EntityListsCopy = InEntityLists;
    int32                        PhaseIdCopy = PhaseId;
    int32                        PlayerIdCopy = PlayerId;
    int32                        PendingACopy = PendingActionA;
    int32                        PendingBCopy = PendingActionB;
    int32                        PendingCCopy = PendingActionC;
    int32                        PendingDCopy = PendingActionD;

    Async(EAsyncExecution::Thread, [this,
        GameStateCopy, PhaseIdCopy, PlayerIdCopy,
        EntityListsCopy,
        PendingACopy, PendingBCopy, PendingCCopy, PendingDCopy]() mutable
        {
            const bool bHasPendingContext = PendingACopy != INDEX_NONE;

            // ----------------------------------------------------------------
            // Extract node and global features from the flat state buffer
            // ----------------------------------------------------------------
            const TArray<float> NodeFeatures = ExtractNodeFeatures(GameStateCopy);
            const TArray<float> GlobalFeatures = ExtractGlobalFeatures(GameStateCopy);

            // ----------------------------------------------------------------
            // Assemble entity tensor
            // ----------------------------------------------------------------
            TArray<float> EntityTensor;
            TArray<float> EntityCounts;
            AssembleEntityTensor(EntityListsCopy, EntityTensor, EntityCounts);

            // ----------------------------------------------------------------
            // Build the batch: the model has a fixed batch size, so the one
            // position is repeated to fill it. Only row 0 is used.
            // ----------------------------------------------------------------
            TArray<TArray<float>> NodeBatch;
            TArray<TArray<float>> GlobalBatch;
            TArray<TArray<float>> EntityBatch;
            TArray<TArray<float>> CountsBatch;
            TArray<float>         PhaseIdsBatch;
            NodeBatch.Init(NodeFeatures, INFERENCE_BATCH_SIZE);
            GlobalBatch.Init(GlobalFeatures, INFERENCE_BATCH_SIZE);
            EntityBatch.Init(EntityTensor, INFERENCE_BATCH_SIZE);
            CountsBatch.Init(EntityCounts, INFERENCE_BATCH_SIZE);
            PhaseIdsBatch.Init((float)PhaseIdCopy, INFERENCE_BATCH_SIZE);

            // ----------------------------------------------------------------
            // Run inference
            // ----------------------------------------------------------------
            TArray<float> OutH2, OutH3, OutH7, OutH10;
            TArray<float> OutH13, OutH14, OutH20, OutH49;
            TArray<float> OutH128, OutH202, OutH330;
            TArray<float> OutH6581, OutH988;
            TArray<float> OutValue;

            const bool bSuccess = RDGCache->RunInference(
                NodeBatch, GlobalBatch, EntityBatch, CountsBatch, PhaseIdsBatch,
                OutH2, OutH3, OutH7, OutH10,
                OutH13, OutH14, OutH20, OutH49,
                OutH128, OutH202, OutH330,
                OutH6581, OutH988,
                OutValue);

            if (!bSuccess)
            {
                UE_LOG(LogTemp, Error,
                    TEXT("GetActionFromAIModel: RunInference failed for PhaseId=%d"),
                    PhaseIdCopy);
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnAIModelActionComplete.Broadcast(-1);
                    });
                return;
            }

            // ----------------------------------------------------------------
            // Select the correct policy output for this phase
            // ----------------------------------------------------------------
            const EPhaseId    TypedPhase = static_cast<EPhaseId>(PhaseIdCopy);
            const EPolicyHead Head = GetPolicyHeadForPhase(TypedPhase);
            const int32       ActionSize = GetPolicySizeForHead(Head);

            const TArray<float>* PolicyOutput = nullptr;
            switch (Head)
            {
            case EPolicyHead::Head2:    PolicyOutput = &OutH2;    break;
            case EPolicyHead::Head3:    PolicyOutput = &OutH3;    break;
            case EPolicyHead::Head7:    PolicyOutput = &OutH7;    break;
            case EPolicyHead::Head10:   PolicyOutput = &OutH10;   break;
            case EPolicyHead::Head13:   PolicyOutput = &OutH13;   break;
            case EPolicyHead::Head14:   PolicyOutput = &OutH14;   break;
            case EPolicyHead::Head20:   PolicyOutput = &OutH20;   break;
            case EPolicyHead::Head49:   PolicyOutput = &OutH49;   break;
            case EPolicyHead::Head128:  PolicyOutput = &OutH128;  break;
            case EPolicyHead::Head202:  PolicyOutput = &OutH202;  break;
            case EPolicyHead::Head330:  PolicyOutput = &OutH330;  break;
            case EPolicyHead::Head6581: PolicyOutput = &OutH6581; break;
            case EPolicyHead::Head988:  PolicyOutput = &OutH988;  break;
            default:
                UE_LOG(LogTemp, Error,
                    TEXT("GetActionFromAIModel: unhandled policy head for PhaseId=%d"),
                    PhaseIdCopy);
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnAIModelActionComplete.Broadcast(-1);
                    });
                return;
            }

            // Row 0 holds this position's policy.
            if (!PolicyOutput || PolicyOutput->Num() < ActionSize)
            {
                UE_LOG(LogTemp, Error,
                    TEXT("GetActionFromAIModel: policy output size mismatch — "
                        "got %d expected %d for PhaseId=%d"),
                    PolicyOutput ? PolicyOutput->Num() : -1,
                    ActionSize, PhaseIdCopy);
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnAIModelActionComplete.Broadcast(-1);
                    });
                return;
            }

            // ----------------------------------------------------------------
            // Get legal action mask
            // ----------------------------------------------------------------
            const FApplyActionResult MaskResult = Internal_GetLegalActionMask(
                GameStateCopy,
                PhaseIdCopy,
                PlayerIdCopy,
                bHasPendingContext,
                PendingACopy,
                PendingBCopy,
                PendingCCopy,
                PendingDCopy,
                INDEX_NONE,
                EntityListsCopy
            );

            if (MaskResult.OutLegalActionMask.Num() != ActionSize)
            {
                AsyncTask(ENamedThreads::GameThread, [this]()
                    {
                        OnAIModelActionComplete.Broadcast(-1);
                    });
                return;
            }

            // ----------------------------------------------------------------
            // Return argmax of masked policy logits
            // ----------------------------------------------------------------
            int32 BestAction = -1;
            float BestLogit = -FLT_MAX;
            int32 LegalCount = 0;

            for (int32 i = 0; i < ActionSize; i++)
            {
                if (!MaskResult.OutLegalActionMask[i])
                    continue;
                LegalCount++;
                const float Logit = (*PolicyOutput)[i];
                if (Logit > BestLogit)
                {
                    BestLogit = Logit;
                    BestAction = i;
                }
            }

            // Marshal result back to game thread
            AsyncTask(ENamedThreads::GameThread, [this, BestAction]()
                {
                    OnAIModelActionComplete.Broadcast(BestAction);
                });
        });

    return -1;
}

TArray<FTerritoryEntityList> UAIManager::DebugSimulateTransition(
    const TArray<float>& InState, int32 InPhaseId, int32 InPlayerId, int32 Action,
    int32 PendingA, int32 PendingB, int32 PendingC, int32 PendingD,
    const TArray<FTerritoryEntityList>& InEntityLists)
{
    return Internal_SimulateTransition(InState, InPhaseId, InPlayerId, Action, true,
        PendingA, PendingB, PendingC, PendingD, INDEX_NONE, InEntityLists).OutEntityLists;
}