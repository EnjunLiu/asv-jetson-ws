from setuptools import find_packages, setup


package_name = "asv_vla"

setup(
    name=package_name,
    version="0.6.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/schema",
            ["schema/frame_record_v1.schema.json"],
        ),
        (
            "share/" + package_name + "/examples",
            ["examples/frame_record_v1.json"],
        ),
    ],
    install_requires=["setuptools", "jsonschema"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Enjun Liu",
    maintainer_email="liuenjun1010@gmail.com",
    description="Fail-closed modular VLA stack for the ASV.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "stub_stack = asv_vla.stub_stack_node:main",
            "stub_stack_without_language = asv_vla.stub_stack_node:main_without_language",
            "safety_tail = asv_vla.stub_stack_node:main_safety_tail",
            "smoke_inputs = asv_vla.smoke_inputs:main",
            "contract_probe = asv_vla.contract_probe:main",
            "language_encoder = asv_vla.language_encoder_node:main",
            "language_embedding_probe = asv_vla.language_embedding_probe:main",
            "visual_encoder = asv_vla.visual_encoder_node:main",
            "visual_encoder_probe = asv_vla.visual_encoder_probe:main",
            "task_entity_tensor = asv_vla.task_entity_tensor_node:main",
            "task_entity_probe = asv_vla.task_entity_probe:main",
            "evaluate_language_similarity = asv_vla.evaluate_language_similarity:main",
            (
                "generate_language_interventions = "
                "asv_vla.generate_language_interventions:main"
            ),
            "evaluate_language_coverage = asv_vla.evaluate_language_coverage:main",
            "validate_frame_record = asv_vla.frame_record:main",
            "record_episode = asv_vla.episode_recorder_node:main",
            "replay_episode = asv_vla.episode_replay_node:main",
            "evaluate_episode = asv_vla.episode:main",
            "day8_replay_probe = asv_vla.day8_replay_probe:main",
            "expert_trajectory = asv_vla.expert_trajectory_node:main",
            (
                "expert_kinematic_executor = "
                "asv_vla.expert_kinematic_executor_node:main"
            ),
            (
                "expert_trajectory_probe = "
                "asv_vla.expert_trajectory_probe:main"
            ),
            (
                "evaluate_expert_labels = "
                "asv_vla.evaluate_expert_labels:main"
            ),
            (
                "build_supervised_dataset = "
                "asv_vla.supervised_dataset:build_main"
            ),
            (
                "evaluate_supervised_dataset = "
                "asv_vla.supervised_dataset:evaluate_main"
            ),
        ],
    },
)
