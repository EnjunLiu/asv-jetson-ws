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
        // Legacy demo layout: the red target is NEAREST (6 m ahead) so a
        // "follow red at 3 m" instruction yields a clean forward approach
        // with the other targets far away (9 m) — the collision margin
        // stays clear along the whole approach path.
        OutLocations.Add(TEXT("target_red"), FVector(600.0, 0.0, 0.0));
        OutLocations.Add(TEXT("target_blue"), FVector(900.0, 0.0, 0.0));
        OutLocations.Add(TEXT("target_left"), FVector(900.0, -300.0, 0.0));
        OutLocations.Add(TEXT("target_right"), FVector(900.0, 300.0, 0.0));
        return true;
    }
    if (LayoutId == TEXT("L6"))
    {
        // Sine-formation demo layout (motion S2): the red/blue pair starts
        // 25 m ahead, side by side with 6 m separation (red left, blue
        // right), and the two white boats sit 35 m ahead on either side as
        // distractors.  The follower approaches from behind and must select
        // the commanded colour while both boats are still in the 90 deg FOV.
        OutLocations.Add(TEXT("target_red"), FVector(2500.0, -300.0, 0.0));
        OutLocations.Add(TEXT("target_blue"), FVector(2500.0, 300.0, 0.0));
        OutLocations.Add(TEXT("target_left"), FVector(3500.0, -800.0, 0.0));
        OutLocations.Add(TEXT("target_right"), FVector(3500.0, 800.0, 0.0));
        return true;
    }
    if (LayoutId == TEXT("L6B"))
    {
        // Mirror of L6: red on the right, blue on the left.  Provides the
        // mirrored geometry needed to verify colour selection both ways.
        OutLocations.Add(TEXT("target_red"), FVector(2500.0, 300.0, 0.0));
        OutLocations.Add(TEXT("target_blue"), FVector(2500.0, -300.0, 0.0));
        OutLocations.Add(TEXT("target_left"), FVector(3500.0, -800.0, 0.0));
        OutLocations.Add(TEXT("target_right"), FVector(3500.0, 800.0, 0.0));
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

TMap<FName, FSceneSineParams> MakeSineParams(
    const FString& MotionState,
    const FString& LayoutId,
    int32 Seed,
    double ForwardSpeedCmPerSec,
    double PeakAmplitudeCm)
{
    TMap<FName, FSceneSineParams> Result;
    if (MotionState != TEXT("S2"))
    {
        return Result;
    }
    // Phase and direction derive from the seed so runs vary while the
    // formation stays deterministic per seed.
    const double Direction = (Seed % 2 == 0) ? 1.0 : -1.0;
    const double PhaseRad =
        Direction * (double)(Seed % 360) * PI / 180.0;
    // The red/blue pair rides the sine on opposite sides of the center line
    // (6 m separation).  Each boat swings with half the peak amplitude so
    // the formation's total lateral extent stays within the configured
    // peak.  The white boats advance straight ahead as distractors (zero
    // lateral amplitude).  Layout L6 puts red left of blue; the mirrored
    // L6B swaps the lateral offsets so the sine swings match the layout.
    const bool RedOnLeft = (LayoutId == TEXT("L6"));
    FSceneSineParams Red;
    Red.ForwardSpeedCmPerSec = ForwardSpeedCmPerSec;
    Red.LateralOffsetCm = RedOnLeft ? -300.0 : 300.0;
    Red.AmplitudeCm = PeakAmplitudeCm / 2.0;
    Red.PhaseRad = PhaseRad;
    Result.Add(TEXT("target_red"), Red);

    FSceneSineParams Blue = Red;
    Blue.LateralOffsetCm = RedOnLeft ? 300.0 : -300.0;
    Result.Add(TEXT("target_blue"), Blue);

    FSceneSineParams White = Red;
    White.LateralOffsetCm = 0.0;
    White.AmplitudeCm = 0.0;
    Result.Add(TEXT("target_left"), White);
    Result.Add(TEXT("target_right"), White);
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
    if (MotionState == TEXT("S2"))
    {
        UE_LOG(
            LogSceneAutomation,
            Display,
            TEXT("SCENE_SINE_PARAMS wavelength_cm=%.0f amplitude_cm=%.0f speed_cm_s=%.0f"),
            SineWavelengthCm,
            SineAmplitudeCm,
            SineSpeedCmPerSec);
    }
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
    // Optional sine-formation parameters (motion S2); defaults are the
    // demo configuration (wavelength 60 m, peak amplitude 6 m, 0.6 m/s).
    FParse::Value(CommandLine, TEXT("SineWavelength="), SineWavelengthCm);
    FParse::Value(CommandLine, TEXT("SineAmplitude="), SineAmplitudeCm);
    FParse::Value(CommandLine, TEXT("SineSpeed="), SineSpeedCmPerSec);
    bYawFixWholeRun = FParse::Param(FCommandLine::Get(), TEXT("YawFixWholeRun"));
    if (SlotId.IsEmpty()
        || !LayoutId.StartsWith(TEXT("L"))
        || (MotionState != TEXT("S0")
            && MotionState != TEXT("S1")
            && MotionState != TEXT("S2"))
        || SceneSeed < 1
        || MaxRuntimeSeconds < 10.0F
        || SineWavelengthCm < 1000.0
        || SineAmplitudeCm < 100.0
        || SineSpeedCmPerSec <= 0.0)
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
    SineParams = MakeSineParams(
        MotionState, LayoutId, SceneSeed, SineSpeedCmPerSec, SineAmplitudeCm);

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
    if (ElapsedSeconds < kAsvYawFixWindowSec || bYawFixWholeRun)
    {
        // Undo seed-driven ASV rotation(s) applied at/after BeginPlay.  The
        // Connection blueprint may spawn its own BP_ASV after ConfigureScene,
        // so fix every BP_ASV_C in the world.  Stops once kinematic setpoints
        // may move the ship, so it does not fight the executor.  With
        // YawFixWholeRun the fix persists to suppress the blueprint's
        // mid-run 180 deg flip (observed under a setpoint stream).
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
        FVector Location;
        if (const FSceneSineParams* Params = SineParams.Find(Binding.EntityId))
        {
            // Analytic world-frame sine: the formation advances along X and
            // each boat oscillates laterally about its center-line offset.
            const double X =
                InitialWorldLocations[Binding.EntityId].X
                + Params->ForwardSpeedCmPerSec * ElapsedSeconds;
            const double Y =
                InitialWorldLocations[Binding.EntityId].Y
                + Params->LateralOffsetCm
                + Params->AmplitudeCm
                    * FMath::Sin(
                        2.0 * PI * Params->ForwardSpeedCmPerSec
                            * ElapsedSeconds / SineWavelengthCm
                        + Params->PhaseRad);
            Location = FVector(X, Y, InitialWorldLocations[Binding.EntityId].Z);
        }
        else
        {
            Location =
                InitialWorldLocations[Binding.EntityId]
                + WorldVelocitiesCmPerSecond[Binding.EntityId] * ElapsedSeconds;
        }
        Target->SetActorLocationAndRotation(
            Location,
            InitialWorldRotations[Binding.EntityId],
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }
    // Diagnostics: sample world positions once per second to verify target
    // motion and to detect any blueprint-driven ASV drift.
    if (FMath::FloorToInt(ElapsedSeconds) != LastPosSampleSecond)
    {
        LastPosSampleSecond = FMath::FloorToInt(ElapsedSeconds);
        for (const FTargetBinding& Binding : TargetBindings)
        {
            AActor* Target = TargetActors[Binding.EntityId].Get();
            if (Target == nullptr)
            {
                continue;
            }
            UE_LOG(
                LogSceneAutomation,
                Display,
                TEXT("SCENE_TARGET_POS t=%.1f entity=%s world=%s"),
                ElapsedSeconds,
                *Binding.EntityId.ToString(),
                *Target->GetActorLocation().ToString());
        }
        if (AActor* Asv = AsvActor.Get())
        {
            UE_LOG(
                LogSceneAutomation,
                Display,
                TEXT("SCENE_ASV_POS t=%.1f world=%s yaw=%d"),
                ElapsedSeconds,
                *Asv->GetActorLocation().ToString(),
                (int32)FMath::RoundToInt(Asv->GetActorRotation().Yaw));
        }
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
