#include "Day12AutomationSubsystem.h"

#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HAL/PlatformMisc.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "UObject/UnrealType.h"

DEFINE_LOG_CATEGORY_STATIC(LogDay12Automation, Log, All);

namespace
{
struct FTargetBinding
{
    FName EntityId;
    const TCHAR* BlueprintClassName;
};

const FTargetBinding TargetBindings[] = {
    {TEXT("target_red"), TEXT("BP_Target_C")},
    {TEXT("target_blue"), TEXT("BP_Target1_C")},
    {TEXT("target_left"), TEXT("BP_Target2_C")},
    {TEXT("target_right"), TEXT("BP_Target3_C")},
};

FString NormalizePropertyName(FString Value)
{
    Value.ReplaceInline(TEXT("_"), TEXT(""));
    Value.ReplaceInline(TEXT(" "), TEXT(""));
    Value.ToLowerInline();
    return Value;
}

bool MakeLayout(const FString& LayoutId, TMap<FName, FVector>& OutLocations)
{
    OutLocations.Reset();
    if (LayoutId == TEXT("L1"))
    {
        OutLocations.Add(TEXT("target_red"), FVector(150.0, 0.0, 0.0));
        OutLocations.Add(TEXT("target_blue"), FVector(400.0, 0.0, 0.0));
        OutLocations.Add(TEXT("target_left"), FVector(250.0, -150.0, 0.0));
        OutLocations.Add(TEXT("target_right"), FVector(250.0, 150.0, 0.0));
        return true;
    }
    if (LayoutId == TEXT("L2"))
    {
        OutLocations.Add(TEXT("target_red"), FVector(400.0, 0.0, 0.0));
        OutLocations.Add(TEXT("target_blue"), FVector(150.0, 0.0, 0.0));
        OutLocations.Add(TEXT("target_left"), FVector(250.0, -150.0, 0.0));
        OutLocations.Add(TEXT("target_right"), FVector(250.0, 150.0, 0.0));
        return true;
    }
    if (LayoutId == TEXT("L3"))
    {
        OutLocations.Add(TEXT("target_red"), FVector(250.0, -120.0, 0.0));
        OutLocations.Add(TEXT("target_blue"), FVector(250.0, 120.0, 0.0));
        OutLocations.Add(TEXT("target_left"), FVector(150.0, -180.0, 0.0));
        OutLocations.Add(TEXT("target_right"), FVector(400.0, 180.0, 0.0));
        return true;
    }
    if (LayoutId == TEXT("L4"))
    {
        OutLocations.Add(TEXT("target_red"), FVector(250.0, 120.0, 0.0));
        OutLocations.Add(TEXT("target_blue"), FVector(250.0, -120.0, 0.0));
        OutLocations.Add(TEXT("target_left"), FVector(400.0, -180.0, 0.0));
        OutLocations.Add(TEXT("target_right"), FVector(150.0, 180.0, 0.0));
        return true;
    }
    return false;
}

TMap<FName, FVector> MakeLocalVelocities(const FString& MotionState, int32 Seed)
{
    TMap<FName, FVector> Result;
    if (MotionState == TEXT("S0"))
    {
        for (const FTargetBinding& Binding : TargetBindings)
        {
            Result.Add(Binding.EntityId, FVector::ZeroVector);
        }
        return Result;
    }

    // Centimetres per second. Pairwise differences remain observable after
    // subtracting the ego velocity in the UE -> ROS relative-velocity field.
    const double Direction = (Seed % 2 == 0) ? 1.0 : -1.0;
    Result.Add(TEXT("target_red"), FVector(0.0, 12.0 * Direction, 0.0));
    Result.Add(TEXT("target_blue"), FVector(0.0, -9.0 * Direction, 0.0));
    Result.Add(TEXT("target_left"), FVector(7.0 * Direction, 3.0, 0.0));
    Result.Add(TEXT("target_right"), FVector(-5.0 * Direction, -4.0, 0.0));
    return Result;
}
} // namespace

bool UDay12AutomationSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
    if (!FParse::Param(FCommandLine::Get(), TEXT("Day12Auto")))
    {
        return false;
    }
    const UWorld* World = Cast<UWorld>(Outer);
    return World != nullptr
        && (World->WorldType == EWorldType::Game
            || World->WorldType == EWorldType::PIE);
}

void UDay12AutomationSubsystem::OnWorldBeginPlay(UWorld& InWorld)
{
    Super::OnWorldBeginPlay(InWorld);
    if (!ReadCommandLine() || !ConfigureScene())
    {
        return;
    }
    bConfigured = true;
    UE_LOG(
        LogDay12Automation,
        Display,
        TEXT("DAY12_UE_READY slot=%s layout=%s motion=%s scene_seed=%d"),
        *SlotId,
        *LayoutId,
        *MotionState,
        SceneSeed);
}

bool UDay12AutomationSubsystem::ReadCommandLine()
{
    const TCHAR* CommandLine = FCommandLine::Get();
    if (!FParse::Value(CommandLine, TEXT("Day12Slot="), SlotId)
        || !FParse::Value(CommandLine, TEXT("Day12Layout="), LayoutId)
        || !FParse::Value(CommandLine, TEXT("Day12Motion="), MotionState)
        || !FParse::Value(CommandLine, TEXT("Day12Seed="), SceneSeed))
    {
        FailAndExit(TEXT("missing slot/layout/motion/seed command-line argument"));
        return false;
    }
    FParse::Value(
        CommandLine,
        TEXT("Day12MaxRuntimeSeconds="),
        MaxRuntimeSeconds);
    if (SlotId.IsEmpty()
        || !LayoutId.StartsWith(TEXT("L"))
        || (MotionState != TEXT("S0") && MotionState != TEXT("S1"))
        || SceneSeed < 1
        || MaxRuntimeSeconds < 10.0F)
    {
        FailAndExit(TEXT("invalid slot/layout/motion/seed/runtime argument"));
        return false;
    }
    return true;
}

AActor* UDay12AutomationSubsystem::FindUniqueActorByClassName(
    const TCHAR* ClassName) const
{
    AActor* Match = nullptr;
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        if (It->GetClass()->GetName() != ClassName)
        {
            continue;
        }
        if (Match != nullptr)
        {
            UE_LOG(
                LogDay12Automation,
                Error,
                TEXT("Duplicate actor class required by Day12 automation: %s"),
                ClassName);
            return nullptr;
        }
        Match = *It;
    }
    return Match;
}

bool UDay12AutomationSubsystem::SetBlueprintInteger(
    AActor& Actor,
    const FString& LogicalName,
    int64 Value) const
{
    const FString Wanted = NormalizePropertyName(LogicalName);
    for (TFieldIterator<FProperty> It(Actor.GetClass()); It; ++It)
    {
        FProperty* Property = *It;
        if (NormalizePropertyName(Property->GetName()) != Wanted)
        {
            continue;
        }
        FNumericProperty* Numeric = CastField<FNumericProperty>(Property);
        if (Numeric == nullptr || !Numeric->IsInteger())
        {
            UE_LOG(
                LogDay12Automation,
                Error,
                TEXT("%s.%s exists but is not an integer"),
                *Actor.GetName(),
                *Property->GetName());
            return false;
        }
        void* ValueAddress = Property->ContainerPtrToValuePtr<void>(&Actor);
        Numeric->SetIntPropertyValue(ValueAddress, Value);
        return true;
    }
    UE_LOG(
        LogDay12Automation,
        Error,
        TEXT("%s has no Blueprint integer property matching %s"),
        *Actor.GetName(),
        *LogicalName);
    return false;
}

bool UDay12AutomationSubsystem::ConfigureScene()
{
    AActor* Asv = FindUniqueActorByClassName(TEXT("BP_ASV_C"));
    AActor* Connection = FindUniqueActorByClassName(TEXT("Connection_C"));
    if (Asv == nullptr || Connection == nullptr)
    {
        FailAndExit(TEXT("BP_ASV_C or Connection_C is missing/duplicated"));
        return false;
    }
    if (!SetBlueprintInteger(*Connection, TEXT("SceneSeed"), SceneSeed))
    {
        FailAndExit(TEXT("cannot set Connection.SceneSeed before BeginPlay"));
        return false;
    }

    TMap<FName, FVector> LocalLocations;
    if (!MakeLayout(LayoutId, LocalLocations))
    {
        FailAndExit(FString::Printf(TEXT("unsupported layout %s"), *LayoutId));
        return false;
    }

    TargetActors.Reset();
    for (const FTargetBinding& Binding : TargetBindings)
    {
        AActor* Target = FindUniqueActorByClassName(Binding.BlueprintClassName);
        if (Target == nullptr)
        {
            FailAndExit(
                FString::Printf(
                    TEXT("required target class %s is missing/duplicated"),
                    Binding.BlueprintClassName));
            return false;
        }
        TargetActors.Add(Binding.EntityId, Target);
    }

    FRandomStream Random(SceneSeed);
    const FTransform AsvTransform = Asv->GetActorTransform();
    const TMap<FName, FVector> LocalVelocities =
        MakeLocalVelocities(MotionState, SceneSeed);

    for (const FTargetBinding& Binding : TargetBindings)
    {
        AActor* Target = TargetActors[Binding.EntityId].Get();
        check(Target != nullptr);

        FVector Local = LocalLocations[Binding.EntityId];
        // Seed-dependent but relation-preserving nuisance variation.
        Local.X += Random.FRandRange(-15.0F, 15.0F);
        Local.Y += Random.FRandRange(-15.0F, 15.0F);

        FVector WorldLocation = AsvTransform.TransformPositionNoScale(Local);
        WorldLocation.Z = Target->GetActorLocation().Z;
        const FVector WorldVelocity = AsvTransform.TransformVectorNoScale(
            LocalVelocities[Binding.EntityId]);

        InitialWorldLocations.Add(Binding.EntityId, WorldLocation);
        InitialWorldRotations.Add(Binding.EntityId, Target->GetActorRotation());
        WorldVelocitiesCmPerSecond.Add(Binding.EntityId, WorldVelocity);
        Target->SetActorLocationAndRotation(
            WorldLocation,
            Target->GetActorRotation(),
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }
    return true;
}

void UDay12AutomationSubsystem::Tick(float DeltaTime)
{
    if (!bConfigured || bExitRequested)
    {
        return;
    }
    ElapsedSeconds += DeltaTime;
    for (const FTargetBinding& Binding : TargetBindings)
    {
        AActor* Target = TargetActors[Binding.EntityId].Get();
        if (Target == nullptr)
        {
            FailAndExit(
                FString::Printf(
                    TEXT("target actor disappeared: %s"),
                    *Binding.EntityId.ToString()));
            return;
        }
        const FVector Location =
            InitialWorldLocations[Binding.EntityId]
            + WorldVelocitiesCmPerSecond[Binding.EntityId] * ElapsedSeconds;
        Target->SetActorLocationAndRotation(
            Location,
            InitialWorldRotations[Binding.EntityId],
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }
    if (ElapsedSeconds >= MaxRuntimeSeconds)
    {
        FailAndExit(TEXT("maximum automation runtime exceeded"));
    }
}

TStatId UDay12AutomationSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(
        UDay12AutomationSubsystem,
        STATGROUP_Tickables);
}

void UDay12AutomationSubsystem::FailAndExit(const FString& Reason)
{
    if (bExitRequested)
    {
        return;
    }
    bExitRequested = true;
    UE_LOG(
        LogDay12Automation,
        Error,
        TEXT("DAY12_UE_FAIL %s"),
        *Reason);
    FPlatformMisc::RequestExit(false);
}
