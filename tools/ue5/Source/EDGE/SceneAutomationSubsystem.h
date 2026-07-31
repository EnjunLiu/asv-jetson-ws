#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"

#include "SceneAutomationSubsystem.generated.h"

class AActor;

/** Per-target motion parameters for the sine formation (motion state S2). */
struct FSceneSineParams
{
    double ForwardSpeedCmPerSec = 0.0;
    double LateralOffsetCm = 0.0;
    double AmplitudeCm = 0.0;
    double PhaseRad = 0.0;
};

/**
 * Command-line-only scene driver.
 *
 * The subsystem is not created during ordinary editor/game runs. Pass
 * -SceneAuto plus the slot/layout/motion/seed arguments to enable it.
 */
UCLASS()
class EDGE_API USceneAutomationSubsystem final
    : public UTickableWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
    virtual void OnWorldBeginPlay(UWorld& InWorld) override;
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;

private:
    bool ReadCommandLine();
    AActor* FindUniqueActorByClassName(const TCHAR* ClassName) const;
    bool SetBlueprintInteger(AActor& Actor, const FString& LogicalName, int64 Value) const;
    bool ConfigureScene();
    void FailAndExit(const FString& Reason);

    FString SlotId;
    FString LayoutId;
    FString MotionState;
    int32 SceneSeed = 0;
    float MaxRuntimeSeconds = 180.0F;
    float ElapsedSeconds = 0.0F;
    bool bConfigured = false;
    bool bExitRequested = false;

    // Motion state S2 (sine formation): all targets advance forward at the
    // same speed while the red/blue pair oscillates laterally about its
    // center line (the two boats ride on opposite sides of the curve) and
    // the white boats run straight ahead as distractors.
    double SineWavelengthCm = 6000.0;
    double SineAmplitudeCm = 600.0;
    double SineSpeedCmPerSec = 60.0;
    TMap<FName, FSceneSineParams> SineParams;

    // Seconds during which Tick re-applies the canonical ASV yaw.  The
    // Connection blueprint consumes SceneSeed at BeginPlay, spawns its own
    // BP_ASV and may randomize the rotation afterwards (e.g. seed 200101 ->
    // yaw 180 deg), which puts the targets behind the camera and makes the
    // visual encoder fail-closed with INVALID_MODALITY.  Re-assert yaw 0 on
    // every BP_ASV_C during startup, then leave the ship alone once
    // kinematic setpoints may arrive.
    static constexpr float kAsvYawFixWindowSec = 8.0F;
    // When set, the yaw fix window is extended to the whole run.  Used to
    // suppress the blueprint's mid-run 180 deg yaw flip (observed at ~35 s
    // under a kinematic setpoint stream) so collection/verification runs
    // stay in the canonical camera-facing frame.
    bool bYawFixWholeRun = false;
    void ForceAsvYawZero(AActor& Asv) const;
    int32 LastYawSampleSecond = -1;
    int32 LastPosSampleSecond = -1;
    TWeakObjectPtr<AActor> AsvActor;

    TMap<FName, TWeakObjectPtr<AActor>> TargetActors;
    TMap<FName, FVector> InitialWorldLocations;
    TMap<FName, FRotator> InitialWorldRotations;
    TMap<FName, FVector> WorldVelocitiesCmPerSecond;
};
