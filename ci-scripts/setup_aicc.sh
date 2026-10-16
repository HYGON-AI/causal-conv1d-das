#!/usr/bin/env bash
set -eu

: "${AICC_NIGHTLY_DIR:?AICC_NIGHTLY_DIR is required}"
: "${ROCM_PATH:?ROCM_PATH is required}"

echo "ROCM_PATH=${ROCM_PATH}"
echo "AICC_NIGHTLY_DIR=${AICC_NIGHTLY_DIR}"

test -d "${AICC_NIGHTLY_DIR}" || {
  echo "ERROR: AICC 目录不存在: ${AICC_NIGHTLY_DIR}" >&2
  exit 1
}

AICC_INSTALLER="$(find "${AICC_NIGHTLY_DIR}" -maxdepth 1 -type f -name 'dtk_llvm_*.run' -print | sort -V | tail -n 1)"
test -n "${AICC_INSTALLER}" || {
  echo "ERROR: ${AICC_NIGHTLY_DIR} 中没有 dtk_llvm_*.run" >&2
  exit 1
}

echo "Install AICC: ${AICC_INSTALLER}"
cp -f "${AICC_INSTALLER}" /tmp/causal-aicc-installer.run
chmod +x /tmp/causal-aicc-installer.run

set +o pipefail
yes | /tmp/causal-aicc-installer.run
AICC_STATUS="${PIPESTATUS[1]}"
rm -f /tmp/causal-aicc-installer.run
test "${AICC_STATUS}" -eq 0

AICC_REAL="$(find /opt/dtk -type f -name aicc -perm -111 -print 2>/dev/null | head -n 1)"
test -n "${AICC_REAL}" || {
  echo "ERROR: AICC 安装完成，但 /opt/dtk 下没有找到可执行 aicc" >&2
  find /opt/dtk -maxdepth 4 -type f \( -name 'aicc' -o -name 'hipcc' \) -print 2>/dev/null || true
  exit 1
}

AILLVM_REAL="$(find /opt/dtk -maxdepth 2 -type d -name aillvm -print 2>/dev/null | head -n 1)"
test -n "${AILLVM_REAL}" || {
  echo "ERROR: AICC 安装完成，但 /opt/dtk 下没有找到 aillvm 目录" >&2
  find /opt/dtk -maxdepth 3 -type d -print 2>/dev/null || true
  exit 1
}

test -x "${AILLVM_REAL}/bin/clang++" || {
  echo "ERROR: AILLVM 中没有可执行 clang++: ${AILLVM_REAL}/bin/clang++" >&2
  find "${AILLVM_REAL}" -maxdepth 3 -type f -name 'clang*' -print 2>/dev/null || true
  exit 1
}

echo "AICC_REAL=${AICC_REAL}"
echo "AILLVM_REAL=${AILLVM_REAL}"

rm -rf "${ROCM_PATH}/aillvm"
ln -s "${AILLVM_REAL}" "${ROCM_PATH}/aillvm"

mkdir -p "${ROCM_PATH}/bin"
printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "${AICC_REAL}" > "${ROCM_PATH}/bin/aicc"
chmod +x "${ROCM_PATH}/bin/aicc"

test -x "${ROCM_PATH}/bin/aicc"
test -x "${ROCM_PATH}/aillvm/bin/clang++"

echo "AICC shim=${ROCM_PATH}/bin/aicc"
echo "AILLVM shim=${ROCM_PATH}/aillvm -> $(readlink -f "${ROCM_PATH}/aillvm")"
"${ROCM_PATH}/bin/aicc" --version