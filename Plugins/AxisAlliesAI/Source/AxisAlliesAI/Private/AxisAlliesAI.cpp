// Copyright Epic Games, Inc. All Rights Reserved.

#include "AxisAlliesAI.h"
#include "UAIManager.h"
#include "MCTSNode.h"
#include "AIInferenceSpec.h"
#include "UAI_ReplayBufferManager.h"
#include "UAI_RDGCache.h"

#define LOCTEXT_NAMESPACE "FAxisAlliesAIModule"

void FAxisAlliesAIModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
}

void FAxisAlliesAIModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FAxisAlliesAIModule, AxisAlliesAI)