"""Exercise one feeder chunk and STOP without cameras, server, or robot."""

import json
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import stream_policy_chunks as feeder
from stream_protocol import COMMAND_PACKET, StreamState


def state(step=0, sequence=0, stop=0, timestamp=None, plan_steps=None):
    return StreamState(time.monotonic_ns() if timestamp is None else timestamp,
                       np.zeros(7), np.zeros(7), np.zeros(3), sequence,
                       step, (4 if sequence else 0) if plan_steps is None else plan_steps, 0, stop)


class FakeMonitor:
    accepted_steps = 4
    def __init__(self, *_args):
        pass

    def wait_for(self, predicate, _timeout, description):
        if description == "the first streamer state":
            result = state()
        elif description == "a post-image robot state":
            result = state()
        elif description == "sequence 1 activation":
            result = state(sequence=1, plan_steps=self.accepted_steps)
        elif description == "final requested plan completion":
            result = state(step=self.accepted_steps, sequence=1, plan_steps=self.accepted_steps)
        elif description == "streamer STOP acknowledgement":
            result = state(step=self.accepted_steps, sequence=1, stop=1, plan_steps=self.accepted_steps)
        else:
            raise AssertionError(description)
        assert predicate(result), description
        return result

    def latest(self):
        return None

    def close(self):
        pass


class FakeCamera:
    def stop(self):
        pass


class FakeSocket:
    packets = []

    def sendto(self, payload, _target):
        self.packets.append(COMMAND_PACKET.unpack(payload))

    def close(self):
        pass


class FakePolicy:
    def __init__(self, *_args):
        pass

    def get_server_metadata(self):
        return {}

    def infer(self, _observation):
        return {"actions": np.zeros((15, 8), dtype=np.float64)}


class FailingPolicy(FakePolicy):
    def infer(self, _observation):
        raise ConnectionError("offline simulated inference failure")


class FeederBoundedTest(unittest.TestCase):
    def check_one_chunk(self, accepted_steps):
        original_int = signal.getsignal(signal.SIGINT)
        original_term = signal.getsignal(signal.SIGTERM)
        FakeSocket.packets = []
        FakeMonitor.accepted_steps = accepted_steps
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 mock.patch.object(feeder.socket, "socket", return_value=FakeSocket()), \
                 mock.patch.object(feeder, "StateMonitor", FakeMonitor), \
                 mock.patch.object(feeder, "start_camera", return_value=FakeCamera()), \
                 mock.patch.object(feeder, "read_rgb", return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
                 mock.patch.object(feeder.websocket_client_policy, "WebsocketClientPolicy", FakePolicy), \
                 mock.patch.object(sys, "argv", ["feeder", "--max-chunks", "1", "--execute-steps", "4",
                                              "--replan-at-step", "2", "--log-dir", directory, "--run-id", "offline"]):
                with self.assertRaises(SystemExit) as outcome:
                    feeder.main()
                self.assertEqual(outcome.exception.code, 0)
                events = [json.loads(line)["event"] for line in
                          (Path(directory) / "offline/events.jsonl").read_text().splitlines()]
                self.assertIn("plan_activated", events)
                self.assertIn("stop_acknowledged", events)
                if accepted_steps < 4:
                    self.assertIn("plan_truncated_by_guard", events)
                self.assertEqual(len(FakeSocket.packets), 2)
                self.assertEqual((FakeSocket.packets[0][4], FakeSocket.packets[0][5]), (0, 4))
                self.assertEqual((FakeSocket.packets[1][4], FakeSocket.packets[1][5]), (1, 0))
        finally:
            signal.signal(signal.SIGINT, original_int)
            signal.signal(signal.SIGTERM, original_term)

    def test_full_four_step_chunk(self):
        self.check_one_chunk(4)

    def test_truncated_single_chunk_stops_at_accepted_length(self):
        self.check_one_chunk(2)

    def test_inference_failure_sends_stop(self):
        original_int = signal.getsignal(signal.SIGINT)
        original_term = signal.getsignal(signal.SIGTERM)
        FakeSocket.packets = []
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 mock.patch.object(feeder.socket, "socket", return_value=FakeSocket()), \
                 mock.patch.object(feeder, "StateMonitor", FakeMonitor), \
                 mock.patch.object(feeder, "start_camera", return_value=FakeCamera()), \
                 mock.patch.object(feeder, "read_rgb", return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
                 mock.patch.object(feeder.websocket_client_policy, "WebsocketClientPolicy", FailingPolicy), \
                 mock.patch.object(sys, "argv", ["feeder", "--max-chunks", "1", "--execute-steps", "4",
                                              "--replan-at-step", "2", "--log-dir", directory, "--run-id", "offline"]):
                with self.assertRaises(SystemExit) as outcome:
                    feeder.main()
                self.assertEqual(outcome.exception.code, feeder.EXIT_INVALID_INFRASTRUCTURE)
                self.assertEqual(len(FakeSocket.packets), 1)
                self.assertEqual(FakeSocket.packets[0][4], 1)
        finally:
            signal.signal(signal.SIGINT, original_int)
            signal.signal(signal.SIGTERM, original_term)


if __name__ == "__main__":
    unittest.main()
