#!/usr/bin/env bash
# 检查 LMCache 和当前目录 vllm/ 是否编译/安装成功。
# 用前先自己 `conda activate opencode`（或装了这两个包的环境）。
set -uo pipefail

# -P: 不把当前目录塞进 sys.path。这里必须加，否则本仓库根下正好有一个
# 同名的 vllm/ 目录，会被当成 namespace package 抢在 editable-install 的
# finder 前面，导致 import vllm 拿到一个空壳（__file__ 是 None）。
python -P - <<'PYEOF'
import importlib.util
import sys
import traceback

REPO_ROOT = "/home/wsh/openhands_code_research"
CHECKS = {
    "vllm": f"{REPO_ROOT}/vllm",
    "lmcache": f"{REPO_ROOT}/LMCache",
}

ok = True

for mod, expected_path in CHECKS.items():
    print(f"\n--- {mod} ---")
    try:
        m = importlib.import_module(mod)
    except Exception:
        print(f"[FAIL] import {mod} 失败:")
        traceback.print_exc()
        ok = False
        continue

    version = getattr(m, "__version__", "?")
    origin = getattr(m, "__file__", None)
    print(f"版本: {version}")
    print(f"来源: {origin}")

    if not origin:
        print(f"[FAIL] {mod} 解析成了空的 namespace package（没有 __file__），装的不对")
        ok = False
    elif expected_path not in origin:
        print(f"[WARN] 来源不在预期目录 {expected_path} 下，可能装的是别的副本")
    else:
        print(f"[OK] 指向本地目录 {expected_path}")

# vllm 编译扩展（最容易因 CXXABI / torch 版本不匹配而报 undefined symbol）
print("\n--- vllm C 扩展 ---")
try:
    import vllm._C_stable_libtorch  # noqa: F401
    print("[OK] vllm._C_stable_libtorch 加载成功")
except Exception:
    print("[FAIL] vllm._C_stable_libtorch 加载失败:")
    traceback.print_exc()
    ok = False

# lmcache 编译扩展
print("\n--- lmcache C 扩展 ---")
for ext in ("c_ops", "native_storage_ops"):
    try:
        importlib.import_module(f"lmcache.{ext}")
        print(f"[OK] lmcache.{ext} 加载成功")
    except Exception:
        print(f"[FAIL] lmcache.{ext} 加载失败:")
        traceback.print_exc()
        ok = False

# vllm 是否能识别到 LMCache connector
print("\n--- vLLM x LMCache 集成 ---")
try:
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    KVConnectorFactory.get_connector_class(
        type("Cfg", (), {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both", "kv_connector_module_path": None})()
    )
    print("[OK] vLLM 能解析 LMCacheConnectorV1")
except Exception:
    print("[FAIL] vLLM 无法解析 LMCacheConnectorV1:")
    traceback.print_exc()
    ok = False

print("\n=====================")
if ok:
    print("全部检查通过：vllm 和 lmcache 都装成功了。")
else:
    print("有检查项失败，看上面的 [FAIL] 定位问题。")
sys.exit(0 if ok else 1)
PYEOF

exit $?
