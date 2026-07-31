#include "SceneAutomationSubsystem.h"

#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HAL/PlatformMisc.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "UObject/UnrealType.h"

DEFINE_LOG_CATEGORY_STATIC(LogSceneAutomation, Log, All);

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
    if (LayoutId == TEXT("L5"))
    {
        // Day 19 demo layout: the red target is NEAREST (6 m ahead) so a
        // "follow red at 3 m" instruction yields a clean forward approach
        // with the other targets far away (9 m) — the collision margin
        // stays clear along the whole approach path.
        OutLocations.Add(TEXT("target_red"), FVector(600.0, 0.0, 0.0));
        OutLocations.Add(TEXT("target_blue"), FVector(900.0, 0.0, 0.0));
        OutLocations.Add(TEXT("target_left"), FVector(900.0, -300.0, 0.0));
        OutLocations.Add(TEXT("target_right"), FVector(900.0, 300.0, 0.0));
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

bool USceneAutomationSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
    if (!FParse::Param(FCommandLine::Get(), TEXT("SceneAuto")))
    {
        return false;
    }
    const UWorld* World = Cast<UWorld>(Outer);
    return World != nullptr
        && (World->WorldType == EWorldType::Game
            || World->WorldType == EWorldType::PIE);
}

void USceneAutomationSubsystem::OnWorldBeginPlay(UWorld& InWorld)
{
    Super::OnWorldBeginPlay(InWorld);
    if (!ReadCommandLine() || !ConfigureScene())
    {
        return;
    }
    bConfigured = true;
    UE_LOG(
        LogSceneAutomation,
        Display,
        TEXT("SCENE_UE_READY slot=%s layout=%s motion=%s scene_seed=%d"),
        *SlotId,
        *LayoutId,
        *MotionState,
        SceneSeed);
}

bool USceneAutomationSubsystem::ReadCommandLine()
{
    const TCHAR* CommandLine = FCommandLine::Get();
    if (!FParse::Value(CommandLine, TEXT("Slot="), SlotId)
        || !FParse::Value(CommandLine, TEXT("Layout="), LayoutId)
        || !FParse::Value(CommandLine, TEXT("Motion="), MotionState)
        || !FParse::Value(CommandLine, TEXT("Seed="), SceneSeed))
    {
        FailAndExit(TEXT("missing slot/layout/motion/seed command-line argument"));
        return false;
    }
    FParse::Value(
        CommandLine,
        TEXT("MaxRuntimeSeconds="),
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

void USceneAutomationSubsystem::ForceAsvYawZero(AActor& Asv) const
{
    Asv.SetActorLocationAndRotation(
        Asv.GetActorLocation(),
        FRotator(0.0, 0.0, 0.0),
        false,
        nullptr,
        ETeleportType::TeleportPhysics);
}

AActor* USceneAutomationSubsystem::FindUniqueActorByClassName(
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
                LogSceneAutomation,
                Error,
                TEXT("Duplicate actor class required by scene automation: %s"),
                ClassName);
            return nullptr;
        }
        Match = *It;
    }
    return Match;
}

bool USceneAutomationSubsystem::SetBlueprintInteger(
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
                LogSceneAutomation,
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
        LogSceneAutomation,
        Error,
        TEXT("%s has no Blueprint integer property matching %s"),
        *Actor.GetName(),
        *LogicalName);
    return false;
}

bool USceneAutomationSubsystem::ConfigureScene()
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

    // Canonical camera-facing pose.  The Connection blueprint consumes
    // SceneSeed and spawns/rotates the ASV afterwards (seed 200101 produced
    // yaw=180 deg, leaving the targets behind the camera).  Force yaw=0
    // before placing targets and re-assert it in Tick (all BP_ASV_C actors)
    // for the first kAsvYawFixWindowSec so the scene is deterministic
    // regardless of BeginPlay ordering.
    ForceAsvYawZero(*Asv);
    AsvActor = Asv;
    UE_LOG(
        LogSceneAutomation,
        Display,
        TEXT("SCENE_ASV_ANCHOR name=%s loc=%s yaw=%d"),
        *Asv->GetName(),
        *Asv->GetActorLocation().ToString(),
        (int32)FMath::RoundToInt(Asv->GetActorRotation().Yaw));

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

void USceneAutomationSubsystem::Tick(float DeltaTime)
{
    if (!bConfigured || bExitRequested)
    {
        return;
    }
    ElapsedSeconds += DeltaTime;
    if (ElapsedSeconds < kAsvYawFixWindowSec)
    {
        // Undo seed-driven ASV rotation(s) applied at/after BeginPlay.  The
        // Connection blueprint may spawn its own BP_ASV after ConfigureScene,
        // so fix every BP_ASV_C in the world.  Stops once kinematic setpoints
        // may move the ship, so it does not fight the executor.
        int32 AsvCount = 0;
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            if (It->GetClass()->GetName() != TEXT("BP_ASV_C"))
            {
                continue;
            }
            ForceAsvYawZero(**It);
            ++AsvCount;
        }
        // Diagnostics: sample the world state once per second while fixing.
        if (FMath::FloorToInt(ElapsedSeconds) != LastYawSampleSecond)
        {
            LastYawSampleSecond = FMath::FloorToInt(ElapsedSeconds);
            UE_LOG(
                LogSceneAutomation,
                Display,
                TEXT("SCENE_YAW_FIX t=%.2f asv_count=%d"),
                ElapsedSeconds,
                AsvCount);
        }
    }
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

TStatId USceneAutomationSubsystem::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(
        USceneAutomationSubsystem,
        STATGROUP_Tickables);
}

void USceneAutomationSubsystem::FailAndExit(const FString& Reason)
{
    if (bExitRequested)
    {
        return;
    }
    bExitRequested = true;
    UE_LOG(
        LogSceneAutomation,
        Error,
        TEXT("SCENE_UE_FAIL %s"),
        *Reason);
    FPlatformMisc::RequestExit(false);
}
