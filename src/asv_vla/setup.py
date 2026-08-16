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
            "image_entity_perception = asv_vla.image_entity_perception_node:main",
            "temporal_entity_tracker = asv_vla.temporal_entity_tracker:main",
            "entity_features = asv_vla.entity_features:main",
            "safety_gate = asv_vla.safety_gate:main",
            "vla_policy = asv_vla.vla_policy_node:main",
            "language_qwen = asv_vla.language_qwen_node:main",
            "task_instruction = asv_vla.task_instruction_node:main",
            "ue_setpoint_adapter = asv_vla.ue_setpoint_adapter:main",
        ],
    },
)
