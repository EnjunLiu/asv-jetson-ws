// EDGE module: scene automation (motion/layout/seed) plus the C++ kinematic
// setpoint executor used for headless closed-loop verification.

using UnrealBuildTool;

public class EDGE : ModuleRules
{
	public EDGE(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"ImageWrapper"
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"RenderCore",
			"Sockets",
			"Networking",
			"Json"
		});
	}
}
