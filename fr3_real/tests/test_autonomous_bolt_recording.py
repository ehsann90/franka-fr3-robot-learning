from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = FR3_ROOT / "model-running" / "record_autonomous_bolt_engagement_demo.py"
SPEC = importlib.util.spec_from_file_location("autonomous_bolt_recording", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AutonomousBoltRecordingTests(unittest.TestCase):
    def test_accelerated_bracket_stages_use_selected_production_profile(self) -> None:
        bracket_config = {"playback_time_scale": 2.222, "binary": "cell-specific"}
        production_config = {"playback_time_scale_by_stage": {"bracket_approach_open": 1.82}, "binary": "4x"}
        self.assertIs(
            MODULE.select_stage_arm_config(
                "bracket_approach_open", bracket_config, production_config, accelerated=True
            ),
            production_config,
        )
        self.assertIs(
            MODULE.select_stage_arm_config(
                "bracket_approach_open", bracket_config, production_config, accelerated=False
            ),
            bracket_config,
        )
        self.assertIs(
            MODULE.select_stage_arm_config(
                "bolt_approach_open", bracket_config, production_config, accelerated=True
            ),
            production_config,
        )

    def test_cell_candidate_is_slowed_only_as_needed_for_profile_caps(self) -> None:
        config = {
            "playback_time_scale": 1.0,
            "playback_time_scale_by_stage": {"bracket_transport_place_closed": 1.0},
            "planning_cap_utilization": 0.9,
            "velocity_cap_rad_s": 1.0,
            "acceleration_cap_rad_s2": 1.0,
            "jerk_cap_rad_s3": 1.0,
        }

        def fake_build(_plan, trial_config, stage_name):
            scale = trial_config["playback_time_scale_by_stage"][stage_name]
            return SimpleNamespace(
                duration_s=scale,
                peak_abs_velocity_rad_s=1.0 / scale,
                peak_abs_acceleration_rad_s2=1.0 / scale**2,
                peak_abs_jerk_rad_s3=1.0 / scale**3,
            )

        with patch.object(MODULE, "build_configured_accelerated_trajectory", side_effect=fake_build):
            trajectory = MODULE.build_cap_limited_accelerated_trajectory(
                object(), config, "bracket_transport_place_closed"
            )
        self.assertGreater(trajectory.duration_s, 1.0)
        self.assertLessEqual(trajectory.peak_abs_velocity_rad_s, 0.9 + 1e-8)
        self.assertLessEqual(trajectory.peak_abs_acceleration_rad_s2, 0.9 + 1e-8)
        self.assertLessEqual(trajectory.peak_abs_jerk_rad_s3, 0.9 + 1e-8)

    def test_bracket_only_endpoint_falls_back_for_generated_manual_task(self) -> None:
        collection = json.loads(
            (FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json").read_text()
        )
        endpoint = MODULE.resolve_success_endpoint(
            bracket_task={"trajectory_dir": "/generated/candidate"},
            full_task=None,
            task=collection["tasks"]["bolt_insert_engage"],
            collection=collection,
            bracket_only=True,
            full_assembly=True,
        )
        self.assertEqual(
            endpoint,
            collection["tasks"]["bracket_placement"]["success_endpoint"],
        )

    def test_pickup_width_fallback_distinguishes_object_from_empty_target(self) -> None:
        config = json.loads(
            (FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json").read_text()
        )["tasks"]["bolt_insert_engage"]["pickup_shaft_grasp"]
        strict, bounded = MODULE.pickup_grasp_evidence(
            config,
            accepted=False,
            is_grasped=False,
            width_m=0.018056,
        )
        self.assertFalse(strict)
        self.assertTrue(bounded)
        _, empty_bounded = MODULE.pickup_grasp_evidence(
            config,
            accepted=False,
            is_grasped=False,
            width_m=config["target_width_m"],
        )
        self.assertFalse(empty_bounded)

    def test_head_obstruction_band_excludes_empty_target(self) -> None:
        config = json.loads(
            (FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json").read_text()
        )["tasks"]["bolt_insert_engage"]["thread_ready_head_grasp"]
        self.assertLess(
            config["target_width_m"],
            config["physical_obstruction_width_min_m"],
        )
        self.assertLessEqual(
            0.018771,
            config["physical_obstruction_width_max_m"],
        )

    def test_failed_two_x_attempt_does_not_consume_completed_trial_allowance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            failed = root / "failed"
            passed = root / "passed"
            failed.mkdir()
            passed.mkdir()
            (failed / "metadata.json").write_text(
                json.dumps({"two_x_profile_enabled": True, "success": False})
            )
            (passed / "metadata.json").write_text(
                json.dumps(
                    {
                        "two_x_profile_enabled": True,
                        "success": False,
                        "guarded_sequence_completed": True,
                    }
                )
            )
            self.assertEqual(MODULE.completed_two_x_episode_count(root), 1)

    def test_full_assembly_uses_passed_profiles_and_bounded_transition(self) -> None:
        config_path = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
        collection, task, preinsert, drop, engagement, commissioning, plans = (
            MODULE.load_inputs(config_path)
        )
        _, bracket_episode, bracket_plans = MODULE.load_bracket_inputs(collection)
        blockers = MODULE.static_blockers(
            collection,
            task,
            preinsert,
            drop,
            engagement,
            commissioning,
            plans,
        )
        blockers.extend(MODULE.object_episode_blockers(bracket_episode, bracket_plans))
        self.assertEqual(blockers, [])
        full = collection["tasks"]["full_bracket_bolt_assembly"]
        self.assertTrue(full["enabled"])
        self.assertEqual(full["segmented_reset_read_only_preview"]["result"], "PASSED")
        self.assertEqual(full["continuous_execution_policy"]["prompts_after_recording_start"], 0)
        self.assertFalse(
            full["continuous_execution_policy"][
                "training_eligibility_before_visual_review"
            ]
        )
        self.assertEqual(full["arm_playback_speed_multiplier"], 1.0)
        self.assertEqual(full["initial_reset_maximum_segments"], 2)
        self.assertTrue(full["two_x_profile"]["enabled"])
        self.assertIsNone(
            full["two_x_profile"]["maximum_supervised_production_trials"]
        )
        self.assertEqual(
            full["two_x_profile"]["offline_validation"],
            "PASSED_ALL_SEVEN_ARM_STAGES_2026-08-14",
        )
        transition_delta = max(
            abs(
                bracket_plans["retreat_open"].expected_stop_q
                - plans["preinsert"]["approach_open"].expected_start_q
            )
        )
        self.assertLessEqual(
            transition_delta,
            full["cross_task_transition"]["maximum_joint_delta_rad"],
        )

    def test_episode_seven_is_training_eligible(self) -> None:
        config_path = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
        collection = json.loads(config_path.read_text())
        accepted = collection["tasks"]["bolt_insert_engage"][
            "accepted_production_episodes"
        ]
        episode = next(row for row in accepted if row["episode_index"] == 7)
        self.assertTrue(episode["bc_training_eligible"])
        self.assertEqual(episode["all_direct_drive_stop_reasons"], "motion_completed")
        self.assertEqual(episode["camera_frame_counts"]["exterior"], 15880)
        self.assertEqual(episode["camera_frame_counts"]["wrist"], 15880)

    def test_trimmed_two_x_full_assembly_profile_is_unlocked_and_bounded(self) -> None:
        config_path = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
        collection, _, _, _, _, _, plans = MODULE.load_inputs(config_path)
        _, _, bracket_plans = MODULE.load_bracket_inputs(collection)
        full = collection["tasks"]["full_bracket_bolt_assembly"]
        commissioning = json.loads(
            (FR3_ROOT / full["two_x_profile"]["commissioning_config"]).read_text()
        )
        self.assertTrue(full["two_x_profile"]["enabled"])
        self.assertTrue(commissioning["final_full_assembly_2x_enabled"])
        self.assertTrue(
            commissioning["threading_2x"]["combined_production_chain_enabled"]
        )
        task = collection["tasks"]["bolt_insert_engage"]
        sequence = task["engagement_drive_sequence"]
        self.assertEqual(
            [stage["angle_deg"] for stage in sequence],
            [85.0, 85.0, 85.0, 50.0, 50.0, 50.0],
        )
        self.assertEqual(task["total_engagement_rotation_deg"], 405.0)
        self.assertFalse(sequence[0]["empty_wrist_ratchet_before"])
        self.assertTrue(all(stage["empty_wrist_ratchet_before"] for stage in sequence[1:]))
        source_plans = {
            "bracket_approach_open": bracket_plans["approach_open"],
            "bracket_transport_place_closed": bracket_plans["transport_place_closed"],
            "bracket_retreat_open": bracket_plans["retreat_open"],
            "bolt_approach_open": plans["preinsert"]["approach_open"],
            "bolt_transport_preinsert_closed": plans["preinsert"]["transport_preinsert_closed"],
            "bolt_lower_held": plans["drop"]["lower_held"],
            "bolt_reposition_regrasp_open": plans["drop"]["reposition_regrasp_open"],
        }
        total_duration = 0.0
        for name, plan in source_plans.items():
            trajectory = MODULE.build_configured_2x_trajectory(plan, commissioning)
            total_duration += trajectory.duration_s
            self.assertLessEqual(
                trajectory.peak_abs_velocity_rad_s,
                commissioning["velocity_cap_rad_s"],
            )
            self.assertLessEqual(
                trajectory.peak_abs_acceleration_rad_s2,
                commissioning["acceleration_cap_rad_s2"],
            )
            self.assertLessEqual(
                trajectory.peak_abs_jerk_rad_s3,
                commissioning["jerk_cap_rad_s3"],
            )
            if name == "bolt_reposition_regrasp_open":
                self.assertTrue(
                    commissioning["stages"][name]["object_aware_physical_trial_passed"]
                )
            else:
                self.assertTrue(
                    commissioning["stages"][name]["empty_hand_physical_trial_passed"]
                )
        self.assertAlmostEqual(total_duration, 95.6, places=1)

    def test_cap_aware_three_x_profile_is_enabled_and_offline_bounded(self) -> None:
        config_path = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
        collection, _, _, _, _, _, plans = MODULE.load_inputs(config_path)
        _, _, bracket_plans = MODULE.load_bracket_inputs(collection)
        full = collection["tasks"]["full_bracket_bolt_assembly"]
        profile = full["three_x_profile"]
        config = json.loads(
            (FR3_ROOT / profile["commissioning_config"]).read_text()
        )
        self.assertTrue(profile["enabled"])
        self.assertTrue(config["final_full_assembly_3x_enabled"])
        self.assertTrue(config["threading_3x"]["combined_production_chain_enabled"])
        self.assertEqual(config["threading_3x"]["peak_angular_speed_rad_s"], 0.09)
        source_plans = {
            "bracket_approach_open": bracket_plans["approach_open"],
            "bracket_transport_place_closed": bracket_plans["transport_place_closed"],
            "bracket_retreat_open": bracket_plans["retreat_open"],
            "bolt_approach_open": plans["preinsert"]["approach_open"],
            "bolt_transport_preinsert_closed": plans["preinsert"]["transport_preinsert_closed"],
            "bolt_lower_held": plans["drop"]["lower_held"],
            "bolt_reposition_regrasp_open": plans["drop"]["reposition_regrasp_open"],
        }
        total_duration = 0.0
        reached_exact_three_x = False
        for name, plan in source_plans.items():
            trajectory = MODULE.build_configured_accelerated_trajectory(
                plan, config, name
            )
            total_duration += trajectory.duration_s
            self.assertLessEqual(
                trajectory.peak_abs_velocity_rad_s,
                0.9 * config["velocity_cap_rad_s"] + 1e-5,
            )
            self.assertLessEqual(
                trajectory.peak_abs_acceleration_rad_s2,
                0.9 * config["acceleration_cap_rad_s2"] + 1e-5,
            )
            self.assertLessEqual(
                trajectory.peak_abs_jerk_rad_s3,
                0.9 * config["jerk_cap_rad_s3"] + 1e-5,
            )
            reached_exact_three_x |= abs(
                config["playback_time_scale_by_stage"][name] - 4.0 / 3.0
            ) < 1e-9
        self.assertTrue(reached_exact_three_x)
        self.assertAlmostEqual(total_duration, 77.864, places=2)

    def test_cap_aware_four_x_profile_retains_force_and_arm_caps(self) -> None:
        config_path = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
        collection, task, _, _, _, _, plans = MODULE.load_inputs(config_path)
        _, _, bracket_plans = MODULE.load_bracket_inputs(collection)
        full = collection["tasks"]["full_bracket_bolt_assembly"]
        profile = full["four_x_profile"]
        config = json.loads(
            (FR3_ROOT / profile["commissioning_config"]).read_text()
        )
        self.assertTrue(profile["enabled"])
        self.assertTrue(config["final_full_assembly_4x_enabled"])
        threading = config["threading_4x"]
        self.assertTrue(threading["combined_production_chain_enabled"])
        self.assertEqual(threading["peak_angular_speed_rad_s"], 0.12)
        self.assertEqual(threading["force_target_N"], 1.200360)
        self.assertEqual(threading["maximum_axial_speed_m_s"], 0.0005)
        self.assertEqual(
            [stage["angle_deg"] for stage in task["engagement_drive_sequence"]],
            [85.0, 85.0, 85.0, 50.0, 50.0, 50.0],
        )
        self.assertEqual(task["total_engagement_rotation_deg"], 405.0)
        source_plans = {
            "bracket_approach_open": bracket_plans["approach_open"],
            "bracket_transport_place_closed": bracket_plans["transport_place_closed"],
            "bracket_retreat_open": bracket_plans["retreat_open"],
            "bolt_approach_open": plans["preinsert"]["approach_open"],
            "bolt_transport_preinsert_closed": plans["preinsert"]["transport_preinsert_closed"],
            "bolt_lower_held": plans["drop"]["lower_held"],
            "bolt_reposition_regrasp_open": plans["drop"]["reposition_regrasp_open"],
        }
        total_duration = 0.0
        for name, plan in source_plans.items():
            trajectory = MODULE.build_configured_accelerated_trajectory(
                plan, config, name
            )
            total_duration += trajectory.duration_s
            self.assertLessEqual(
                trajectory.peak_abs_velocity_rad_s,
                config["velocity_cap_rad_s"],
            )
            self.assertLessEqual(
                trajectory.peak_abs_acceleration_rad_s2,
                config["acceleration_cap_rad_s2"],
            )
            self.assertLessEqual(
                trajectory.peak_abs_jerk_rad_s3,
                config["jerk_cap_rad_s3"],
            )
        self.assertAlmostEqual(total_duration, 76.530, places=2)

    def test_stationary_bridge_fills_handoff_at_requested_rate(self) -> None:
        source = {
            "time_ns": 1_000_000_000,
            "q": [0.1] * 7,
            "dq": [0.2] * 7,
            "dq_d": [0.2] * 7,
            "ddq_d": [0.2] * 7,
        }
        bridge = MODULE.stationary_state_bridge(
            source,
            1_100_000_000,
            30.0,
            0.015746,
            True,
            0.015746,
        )
        self.assertEqual(len(bridge), 3)
        self.assertTrue(all(row["dq"] == [0.0] * 7 for row in bridge))
        self.assertTrue(all(row["gripper_is_grasped"] for row in bridge))
        self.assertEqual(source["dq"], [0.2] * 7)
        self.assertLess(bridge[0]["time_ns"], bridge[1]["time_ns"])

    def test_direct_csv_requires_and_preserves_absolute_time(self) -> None:
        header = (
            "wall_time_ns,time_s,target_force_N,measured_delta_force_N,"
            "requested_speed_m_s,commanded_speed_m_s,axial_travel_m,"
            "lateral_travel_m,maximum_abs_dq_rad_s,"
            "commanded_angular_speed_rad_s,measured_yaw_rad,residual_tz_Nm,"
            + ",".join(f"q{i}" for i in range(7))
            + ","
            + ",".join(f"dq{i}" for i in range(7))
            + ",x,y,z\n"
        )
        values = [
            "123456789",
            "0.001",
            *(["0"] * 10),
            *(["0.1"] * 7),
            *(["0.0"] * 7),
            "0.59",
            "0.32",
            "0.10",
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "direct.csv"
            path.write_text(header + ",".join(values) + "\n")
            samples = MODULE.direct_samples(path, 0.0157, True, 0.015746)
        self.assertEqual(samples[0]["time_ns"], 123456789)
        self.assertEqual(samples[0]["q"], [0.1] * 7)
        self.assertEqual(samples[0]["gripper_command_width"], 0.015746)

    def test_retimed_direct_csv_preserves_state_and_gripper_labels(self) -> None:
        header = (
            "wall_time_ns,time_s,maximum_tracking_error_rad,maximum_abs_dq_rad_s,stopping,"
            + ",".join(f"q{i}" for i in range(7))
            + ","
            + ",".join(f"dq{i}" for i in range(7))
            + ","
            + ",".join(f"q_reference{i}" for i in range(7))
            + ",x,y,z\n"
        )
        values = [
            "123456789",
            "0.001",
            "0.0002",
            "0.01",
            "0",
            *(["0.1"] * 7),
            *(["0.0"] * 7),
            *(["0.1"] * 7),
            "0.59",
            "0.32",
            "0.10",
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "retimed.csv"
            path.write_text(header + ",".join(values) + "\n")
            samples = MODULE.retimed_direct_samples(path, 0.038, True, 0.03)
        self.assertEqual(samples[0]["time_ns"], 123456789)
        self.assertEqual(samples[0]["q"], [0.1] * 7)
        self.assertTrue(samples[0]["gripper_is_grasped"])
        self.assertEqual(
            samples[0]["state_source"], "direct_libfranka_retimed_2x_telemetry"
        )


if __name__ == "__main__":
    unittest.main()
