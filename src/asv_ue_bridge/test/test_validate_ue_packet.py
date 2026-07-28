import copy
import json
import math
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR / "scripts"))

from validate_ue_packet import validate_packet  # noqa: E402


class UePacketValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = json.loads(
            (PACKAGE_DIR / "test" / "data" / "ue_packet_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_checked_in_packet_is_valid(self) -> None:
        self.assertEqual(validate_packet(self.sample), [])

    def test_metadata_placeholders_are_rejected(self) -> None:
        packet = copy.deepcopy(self.sample)
        packet["Run_ID"] = ""
        packet["Scene_Seed"] = "12345"
        packet["Frame_Index"] = -1
        errors = validate_packet(packet)
        self.assertIn("Run_ID must be a non-empty string", errors)
        self.assertIn("Scene_Seed must be an integer", errors)
        self.assertIn("Frame_Index must be a non-negative integer", errors)

    def test_duplicate_entity_id_is_rejected(self) -> None:
        packet = copy.deepcopy(self.sample)
        packet["Entities"].append(copy.deepcopy(packet["Entities"][0]))
        self.assertTrue(
            any("Entity_Id is duplicated" in error for error in validate_packet(packet))
        )

    def test_non_finite_vector_is_rejected(self) -> None:
        packet = copy.deepcopy(self.sample)
        packet["Entities"][0]["RelativePosition"]["x"] = math.nan
        self.assertTrue(
            any(
                "RelativePosition.x must be finite" in error
                for error in validate_packet(packet)
            )
        )

    def test_camera_values_must_be_bytes(self) -> None:
        packet = copy.deepcopy(self.sample)
        packet["Camera_Capture"] = [256]
        self.assertIn(
            "Camera_Capture values must be bytes",
            validate_packet(packet),
        )


if __name__ == "__main__":
    unittest.main()
