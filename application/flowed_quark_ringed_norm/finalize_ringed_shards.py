#!/usr/bin/env python3
"""Finalize complete standalone ringed base shards."""

import argparse

from pyquda_measurement_utils.flowed_quark_ringed_norm import (
    finalize_ringed_quark_1pt_shards,
)


parser = argparse.ArgumentParser()
parser.add_argument("--shard-dir", required=True)
parser.add_argument("--canonical-tag", required=True)
parser.add_argument("--n-base-noise", required=True, type=int)
args = parser.parse_args()

print(finalize_ringed_quark_1pt_shards(
    args.shard_dir, args.canonical_tag, args.n_base_noise
))
