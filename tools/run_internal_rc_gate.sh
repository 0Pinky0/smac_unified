#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${SMAC_UNIFIED_CONDA_ENV:-smacnt}"

echo "[1/7] Core unit/contract tests"
conda run -n "${ENV_NAME}" python "tools/run_core_tests.py"

echo "[2/7] Critical strict parity gate (smac + smacv2)"
conda run -n "${ENV_NAME}" python "tools/native_core_validation.py" \
  --families smac smacv2 \
  --matrix-preset critical-core \
  --profile steady \
  --steady-steps 120 \
  --steady-warmup-steps 20 \
  --steady-parity-mode strict \
  --repeats 2 \
  --assert-parity \
  --output-json "tools/native_core_validation_critical_core.json"

echo "[3/7] smac-hard long-tail strict parity gate (B0)"
conda run -n "${ENV_NAME}" python "tools/native_core_validation.py" \
  --families smac-hard \
  --matrix-preset smac-hard-longtail \
  --profile steady \
  --steady-steps 60 \
  --steady-warmup-steps 10 \
  --steady-parity-mode strict \
  --repeats 1 \
  --repeat-seed-stride 0 \
  --seed 1 \
  --force-opponent-actions-from-bridge \
  --assert-parity \
  --output-json "tools/native_core_validation_smachard_longtail_b0.json"

echo "[4/7] smac-hard long-tail strict parity gate (B1)"
conda run -n "${ENV_NAME}" python "tools/native_core_validation.py" \
  --families smac-hard \
  --matrix-preset smac-hard-longtail \
  --profile steady \
  --steady-steps 60 \
  --steady-warmup-steps 10 \
  --steady-parity-mode strict \
  --repeats 1 \
  --repeat-seed-stride 0 \
  --seed 1 \
  --force-opponent-actions-from-bridge \
  --native-options-json '{"reuse_step_observe_requests": true}' \
  --assert-parity \
  --output-json "tools/native_core_validation_smachard_longtail_b1.json"

echo "[5/7] smac-hard long-tail strict parity gate (B2)"
conda run -n "${ENV_NAME}" python "tools/native_core_validation.py" \
  --families smac-hard \
  --matrix-preset smac-hard-longtail \
  --profile steady \
  --steady-steps 60 \
  --steady-warmup-steps 10 \
  --steady-parity-mode strict \
  --repeats 1 \
  --repeat-seed-stride 0 \
  --seed 1 \
  --force-opponent-actions-from-bridge \
  --native-options-json '{"reuse_step_observe_requests": true, "pipeline_step_and_observe": true}' \
  --assert-parity \
  --output-json "tools/native_core_validation_smachard_longtail_b2.json"

echo "[6/7] smac-hard long-tail strict parity gate (B3)"
conda run -n "${ENV_NAME}" python "tools/native_core_validation.py" \
  --families smac-hard \
  --matrix-preset smac-hard-longtail \
  --profile steady \
  --steady-steps 60 \
  --steady-warmup-steps 10 \
  --steady-parity-mode strict \
  --repeats 1 \
  --repeat-seed-stride 0 \
  --seed 1 \
  --force-opponent-actions-from-bridge \
  --native-options-json '{"reuse_step_observe_requests": true, "pipeline_step_and_observe": true, "pipeline_actions_and_step": true}' \
  --assert-parity \
  --output-json "tools/native_core_validation_smachard_longtail_b3.json"

echo "[7/7] Native-only throughput sanity lane"
conda run -n "${ENV_NAME}" python "tools/native_core_validation.py" \
  --families smac smacv2 smac-hard \
  --profile quick \
  --steps 5 \
  --warmup-steps 1 \
  --bridge-lane off \
  --parallel-envs 4 \
  --pool-mode thread \
  --output-json "tools/native_core_validation_throughput_sanity.json"

echo "Internal RC gate completed successfully."
