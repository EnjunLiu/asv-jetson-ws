#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"

#include "Day12AutomationSubsystem.generated.h"

class AActor;

/**
 * Command-line-only Day 12 scene driver.
 *
 * The subsystem is not created during ordinary editor/game runs. Pass
 * -Day12Auto plus the slot/layout/motion/seed arguments to enable it.
 */
UCLASS()
class EDGE_API UDay12AutomationSubsystem final
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

    TMap<FName, TWeakObjectPtr<AActor>> TargetActors;
    TMap<FName, FVector> InitialWorldLocations;
    TMap<FName, FRotator> InitialWorldRotations;
    TMap<FName, FVector> WorldVelocitiesCmPerSecond;
};
