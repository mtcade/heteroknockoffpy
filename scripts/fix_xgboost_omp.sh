#!/usr/bin/env bash
#
# fix_xgboost_omp.sh
#
# macOS only. xgboost's wheel links against Homebrew's libomp, but torch's wheel
# bundles its own separate libomp.dylib. Loading both into the same process
# crashes (SIGSEGV inside libomp's worker-thread code) the first time a torch
# model and an xgboost model are both used. Re-point xgboost's dylib to load
# torch's already-loaded libomp instead of Homebrew's, so only one loads.
#
# Safe to re-run: if the rpath was already fixed, this is a no-op.

set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
    echo "fix_xgboost_omp.sh: not macOS, nothing to do."
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VIRTUAL_ENV:-${REPO_ROOT}/.venv}"

SITE_PACKAGES="$(find "${VENV_DIR}/lib" -maxdepth 1 -type d -name 'python3.*' 2>/dev/null | head -n1)/site-packages"

TORCH_LIB="${SITE_PACKAGES}/torch/lib"
XGB_DYLIB="${SITE_PACKAGES}/xgboost/lib/libxgboost.dylib"
HOMEBREW_LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)/lib"

if [[ ! -f "${TORCH_LIB}/libomp.dylib" ]]; then
    echo "fix_xgboost_omp.sh: ${TORCH_LIB}/libomp.dylib not found -- is torch installed?" >&2
    exit 1
fi

if [[ ! -f "${XGB_DYLIB}" ]]; then
    echo "fix_xgboost_omp.sh: ${XGB_DYLIB} not found -- is xgboost installed?" >&2
    exit 1
fi

CURRENT_RPATHS="$(otool -l "${XGB_DYLIB}" | awk '/LC_RPATH/{getline; getline; print $2}')"

if grep -qxF "${TORCH_LIB}" <<< "${CURRENT_RPATHS}"; then
    echo "fix_xgboost_omp.sh: already fixed (rpath already points at ${TORCH_LIB})."
    exit 0
fi

if ! grep -qxF "${HOMEBREW_LIBOMP_PREFIX}" <<< "${CURRENT_RPATHS}"; then
    echo "fix_xgboost_omp.sh: expected rpath entry '${HOMEBREW_LIBOMP_PREFIX}' not found on ${XGB_DYLIB};" >&2
    echo "  current rpaths: ${CURRENT_RPATHS}" >&2
    echo "  xgboost's build layout may have changed -- inspect manually." >&2
    exit 1
fi

install_name_tool -rpath "${HOMEBREW_LIBOMP_PREFIX}" "${TORCH_LIB}" "${XGB_DYLIB}"
codesign --force -s - "${XGB_DYLIB}"

echo "fix_xgboost_omp.sh: patched ${XGB_DYLIB} to use torch's bundled libomp.dylib."
