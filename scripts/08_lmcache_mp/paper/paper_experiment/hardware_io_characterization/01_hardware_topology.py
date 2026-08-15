#!/usr/bin/env python3
"""Collect CPU, NUMA, storage, GPU, and PCIe topology without mutation."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any
import re
from common import discover_layers, load_config, metadata_for_layers, numa_nodes, write_test_result


# 清除终端颜色、下划线和光标控制等 ANSI 转义字符
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_output(output: str) -> Any:
    """Clean command output and decode JSON output when possible."""
    output = ANSI_ESCAPE.sub("", output).strip()

    if not output:
        return ""

    # lscpu、lsblk、findmnt 等命令会输出 JSON。
    # 将其转换成真正的 dict/list，避免外层 JSON 出现大量 \n 和 \"。
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        # CSV、表格和普通文本继续保留为字符串
        return output


def run_command(arguments: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        return {
            "status": "unavailable",
            "command": arguments,
            "error": str(error),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "command": arguments,
            "error": str(error),
        }

    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "command": arguments,
        "returncode": completed.returncode,
        "stdout": clean_output(completed.stdout),
        "stderr": clean_output(completed.stderr),
    }


def read_sysfs(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def gpu_pcie_sysfs() -> list[dict[str, Any]]:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )

    if query.get("status") != "ok":
        return []

    stdout = query.get("stdout")
    if not isinstance(stdout, str):
        return []

    rows: list[dict[str, Any]] = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            index_text, bus_text = (
                part.strip() for part in line.split(",", 1)
            )
            domain, bus, device_function = bus_text.lower().split(":")
        except ValueError:
            continue

        # nvidia-smi 可能返回 8 位 PCI domain，例如 00000000:17:00.0；
        # Linux sysfs 通常使用后 4 位，例如 0000:17:00.0。
        if len(domain) == 8:
            domain = domain[-4:]

        address = f"{domain}:{bus}:{device_function}"
        root = Path("/sys/bus/pci/devices") / address

        rows.append(
            {
                "gpu_index": int(index_text),
                "pci_bus_id": bus_text,
                "sysfs_path": str(root),
                "numa_node": read_sysfs(root / "numa_node"),
                "current_link_speed": read_sysfs(
                    root / "current_link_speed"
                ),
                "current_link_width": read_sysfs(
                    root / "current_link_width"
                ),
                "maximum_link_speed": read_sysfs(
                    root / "max_link_speed"
                ),
                "maximum_link_width": read_sysfs(
                    root / "max_link_width"
                ),
            }
        )

    return rows



def main() -> None:
    config = load_config()
    manifest, layers = discover_layers(config)
    pool_path = str(Path(config["skill_cache"]["pool_dir"]).resolve())
    payload = {
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "skill_cache": metadata_for_layers(manifest, layers),
        "numa_cpu_map": {str(node): cpus for node, cpus in numa_nodes().items()},
        "commands": {
            "lscpu": run_command(["lscpu", "--json"]),
            "memory": run_command(["free", "--bytes"]),
            "block_devices": run_command(
                ["lsblk", "--json", "--bytes", "-o", "NAME,TYPE,SIZE,MODEL,ROTA,TRAN,MOUNTPOINTS"]
            ),
            "cache_mount": run_command(["findmnt", "--json", "--target", pool_path]),
            "gpu": run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "gpu_topology": run_command(["nvidia-smi", "topo", "-m"]),
        },
        "gpu_pcie_sysfs": gpu_pcie_sysfs(),
    }
    path = write_test_result("01_hardware_topology", config, payload)
    print(f"[completed] {path}")


if __name__ == "__main__":
    main()
