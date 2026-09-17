// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class AxisAlliesAI : ModuleRules
{
    public AxisAlliesAI(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "InputCore",
                "Json",
                "JsonUtilities",
                "Networking",
                "Sockets",
                "Projects",
                "NNE",
                "NNERuntimeRDG",
                "RHI",
                "RenderCore",
                "Renderer",
            }
        );

        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "CoreUObject",
                "Engine",
                "Slate",
                "SlateCore",
                "NNE",
                "NNERuntimeRDG",
                "RenderCore",
                "RHI",
                "Renderer",
            }
        );

        if (Target.Type == TargetType.Editor)
        {
            PrivateDependencyModuleNames.Add("Kismet");
        }

        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
            }
        );

        PublicDefinitions.Add("AXISALLIESAI_API=DLLEXPORT");
    }
}