from setuptools import find_packages, setup


package_name = "vla"

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
            "image_entity_perception = vla.image_entity_perception_node:main",
            "temporal_entity_tracker = vla.temporal_entity_tracker:main",
            "safety_gate = vla.safety_gate:main",
            "vla_policy = vla.vla_policy_node:main",
            "language_qwen = vla.language_qwen_node:main",
            "task_instruction = vla.task_instruction_node:main",
        ],
    },
)
