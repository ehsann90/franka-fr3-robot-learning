"""Offline checks of the 176-byte C++ state layout and Python decoder."""

import unittest

from stream_protocol import STATE_MAGIC, STATE_PACKET, VERSION, unpack_state


class StateContractTest(unittest.TestCase):
    def test_layout_and_field_mapping(self):
        self.assertEqual(STATE_PACKET.size, 176)
        payload = STATE_PACKET.pack(STATE_MAGIC, VERSION, 123456, *range(17), 42, 2, 4, 1, 0)
        state = unpack_state(payload)
        self.assertEqual(state.monotonic_ns, 123456)
        self.assertEqual(state.q.tolist(), list(range(7)))
        self.assertEqual(state.dq.tolist(), list(range(7, 14)))
        self.assertEqual(state.tcp.tolist(), [14, 15, 16])
        self.assertEqual((state.active_sequence, state.plan_step, state.plan_steps, state.activation_offset),
                         (42, 2, 4, 1))
        self.assertEqual(state.stop_reason_name, "none")
        self.assertIsNone(unpack_state(payload[:-1]))
        self.assertIsNone(unpack_state(STATE_PACKET.pack(STATE_MAGIC, VERSION + 1, 123456,
                                                         *range(17), 42, 2, 4, 1, 0)))


if __name__ == "__main__":
    unittest.main()
