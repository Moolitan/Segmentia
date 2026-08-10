# Hardware I/O characterization

This directory measures the real CSKCache data path with the existing 8K-token
`paper-write` offline Skill KV. It does not start vLLM and never modifies the
offline pool.

## 测试内容

五个独立测试分别为：

1. `01_hardware_topology.py`：采集 CPU、NUMA、SSD、GPU 和 PCIe 拓扑。
2. `02_ssd_to_cpu.py`：读取真实 40 层 Skill KV，测量 cold/warm、1/4 线程的 SSD/page cache→CPU 性能。
3. `03_cpu_memory.py`：测量本地 NUMA、跨 NUMA 和 pageable→pinned 内存拷贝。
4. `04_pcie_h2d.py`：测量 pageable/pinned、同步/异步 PCIe H2D 传输。
5. `05_skill_kv_path.py`：测量真实 SSD/page cache→pinned CPU→GPU 的串行完整路径。

另有 `06_analyze_results.py`，用于汇总五项测试结果并生成论文图表、CSV 和总结文档。

Edit `config.yaml` when changing the run name, cache object, repetitions, or GPU.
No command-line arguments are required.

```bash
conda activate opencode
cd /home/wsh/openhands_code_research/scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization

python 01_hardware_topology.py
python 02_ssd_to_cpu.py
python 03_cpu_memory.py
python 04_pcie_h2d.py
python 05_skill_kv_path.py
python 06_analyze_results.py
```

The five measurements are independent. Re-running one script atomically replaces
only that script's JSON result in the configured run directory. Run
`06_analyze_results.py` after the desired measurements finish.

Raw output:

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/
  paper_experiment_hardware_io/<run.id>/
```

Lightweight paper artifacts:

```text
results/problem_exploration/hardware_io_characterization/
```

`02_ssd_to_cpu.py` uses per-file `POSIX_FADV_DONTNEED` for the cold condition;
it does not write to `/proc/sys/vm/drop_caches`. The complete-path script uses
one pinned layer buffer and one GPU layer buffer, so it measures the serial
SSD→pinned→GPU baseline without allocating a 1.3-GiB staging object.
