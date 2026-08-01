"""Compatibility import for the packaged policy architecture.

The ROS runtime uses :mod:`asv_vla.policy_model` directly so an installed
Jetson package does not depend on a checkout-level ``training`` directory.
Keep this module for existing PC training scripts and tests.

# Compatibility marker retained for the v1 configuration contract:
# entity_attention_mode: str = "legacy"
# language_conditioned_entity_attention: bool = False
"""

try:
    from asv_vla.policy_model import (
        PolicyOutput,
        SmallPolicyConfig,
        SmallTrajectoryPolicy,
    )
except ModuleNotFoundError as exc:
    # A checkout-level training invocation may run without the ROS package
    # installed.  Add only this repository's src root for that PC workflow;
    # the installed ROS runtime never imports this compatibility module.
    if exc.name != "asv_vla":
        raise
    import sys
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1] / "src" / "asv_vla"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from asv_vla.policy_model import (  # noqa: E402
        PolicyOutput,
        SmallPolicyConfig,
        SmallTrajectoryPolicy,
    )

__all__ = ["PolicyOutput", "SmallPolicyConfig", "SmallTrajectoryPolicy"]
