#!/usr/bin/env python3
"""Compute ringed factors only after averaging explicit configuration inputs."""

import argparse

from pyquda_measurement_utils.flowed_quark_ringed_norm import analyze_ringed_ensemble


parser = argparse.ArgumentParser()
parser.add_argument("--input", action="append", required=True, dest="inputs")
parser.add_argument("--output", required=True)
parser.add_argument("--nc", type=int, default=3)
args = parser.parse_args()

print(analyze_ringed_ensemble(args.inputs, args.output, nc=args.nc))
