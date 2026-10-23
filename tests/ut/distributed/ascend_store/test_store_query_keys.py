# SPDX-License-Identifier: Apache-2.0
"""Query keys must match keys written to the pool, including ordering."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, PoolKey


class TestStoreQueryKeys(unittest.TestCase):
    @staticmethod
    def _scheduler(**overrides):
        scheduler = pool_scheduler.KVPoolScheduler.__new__(pool_scheduler.KVPoolScheduler)
        attributes = dict(
            model_name="model@with/分隔符",
            tp_size=4,
            put_step=1,
            dcp_size=2,
            pcp_size=2,
            pp_size=3,
            pp_rank=1,
            num_layers=4,
            pp_layer_offset=7,
            kv_cache_group_families=["default", "c4", "c128"],
        )
        attributes.update(overrides)
        for name, value in attributes.items():
            setattr(scheduler, name, value)
        return scheduler

    @staticmethod
    def _written_keys(scheduler, hashes, include_layers, group_id):
        """Use the writer's key classes as the compatibility oracle."""
        families = scheduler.kv_cache_group_families
        family = families[group_id] if group_id < len(families) else "default"
        ranks = itertools.product(
            range(scheduler.dcp_size),
            range(scheduler.tp_size // scheduler.put_step),
            [scheduler.pp_rank] if include_layers else range(scheduler.pp_size),
        )
        metadata = [
            KeyMetadata(scheduler.model_name, head, dcp, pp, kv_cache_group_id=group_id, cache_family=family)
            for dcp, head, pp in ranks
        ]
        keys_by_block = []
        for block_hash in hashes:
            chunk_hash = block_hash if isinstance(block_hash, str) else block_hash.hex()
            keys = [PoolKey(meta, chunk_hash) for meta in metadata]
            if include_layers:
                keys = [
                    layer_key
                    for key in keys
                    for layer_key in key.split_layers(scheduler.num_layers, scheduler.pp_layer_offset)
                ]
            keys_by_block.append([key.to_string() for key in keys])
        return keys_by_block

    def test_namespace_order_and_hash_encodings(self):
        topologies = [(1, 1, 1, 1, 1), (4, 1, 1, 3, 1), (8, 4, 2, 2, 3), (4, 1, 3, 2, 2)]
        hash_sets = [[], [b"\x00\xff"], [b"", "", "AB@cd", bytearray(b"\x00\xff"), b"\xaa", b"\xaa"]]
        for topology, hashes, layers, group in itertools.product(topologies, hash_sets, [False, True], [0, 1, 3]):
            with self.subTest(topology=topology, hashes=hashes, layers=layers, group=group):
                tp, step, dcp, pp, pcp = topology
                scheduler = self._scheduler(tp_size=tp, put_step=step, dcp_size=dcp, pp_size=pp, pcp_size=pcp)
                actual = scheduler._generate_store_query_keys(iter(hashes), layers, group)
                self.assertEqual(actual, self._written_keys(scheduler, hashes, layers, group))

    def test_no_cross_call_or_result_aliasing(self):
        scheduler = self._scheduler()
        hashes = [b"\xaa", b"\xaa"]
        first = scheduler._generate_store_query_keys(hashes, True, 1)
        first[0].clear()
        self.assertTrue(first[1])
        scheduler.model_name = "another-model"
        scheduler.pp_layer_offset = 40
        scheduler.kv_cache_group_families[1] = "c64"
        scheduler.tp_size = 2
        second = scheduler._generate_store_query_keys(hashes, True, 1)
        self.assertEqual(second, self._written_keys(scheduler, hashes, True, 1))
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(scheduler._generate_store_query_keys([], True), [])

    def test_empty_rank_and_layer_ranges(self):
        for overrides, layers in [
            ({"dcp_size": 0}, False),
            ({"pp_size": 0}, False),
            ({"tp_size": 1, "put_step": 2}, False),
            ({"num_layers": 0}, True),
        ]:
            with self.subTest(overrides=overrides):
                scheduler = self._scheduler(**overrides)
                self.assertEqual(scheduler._generate_store_query_keys([b"a", b"b"], layers), [[], []])

    def test_invalid_configuration_and_hash(self):
        with self.assertRaises(ZeroDivisionError):
            self._scheduler(put_step=0)._generate_store_query_keys([])
        with self.assertRaises(TypeError):
            self._scheduler(pp_size="invalid")._generate_store_query_keys([b"a"])
        with self.assertRaises(AttributeError):
            self._scheduler()._generate_store_query_keys([None])
        with self.assertRaises(IndexError):
            self._scheduler()._generate_store_query_keys([b"a"], kv_cache_group_id=-4)

    def test_metadata_construction_does_not_scale_with_blocks(self):
        scheduler = self._scheduler()
        hashes = [i.to_bytes(32, "big") for i in range(128)]
        for layers in (False, True):
            with (
                self.subTest(layers=layers),
                patch.object(pool_scheduler, "KeyMetadata", wraps=KeyMetadata) as metadata,
            ):
                actual = scheduler._generate_store_query_keys(hashes, layers)
                rank_count = scheduler.dcp_size * (scheduler.tp_size // scheduler.put_step)
                rank_count *= 1 if layers else scheduler.pp_size
                self.assertLessEqual(metadata.call_count, rank_count)
                self.assertEqual(actual, self._written_keys(scheduler, hashes, layers, 0))

    def test_lookup_prefix_hit_stops_at_first_missing_rank_or_layer(self):
        scheduler = self._scheduler()
        scheduler._block_size = 16
        scheduler.use_layerwise = True
        request = SimpleNamespace(request_id="query", block_hashes=[b"a", b"b", b"c"])
        keys_by_block = self._written_keys(scheduler, request.block_hashes, True, 0)
        stored = {key for keys in keys_by_block for key in keys}
        stored.remove(keys_by_block[1][-1])
        scheduler.store_scheduler = SimpleNamespace(batch_is_exist=lambda keys: [int(key in stored) for key in keys])
        self.assertEqual(scheduler._get_store_lookup_hit_tokens(request, 48, 0, True), 16)
