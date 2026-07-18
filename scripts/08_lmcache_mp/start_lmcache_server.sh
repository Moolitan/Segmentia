#!/usr/bin/env bash
# 第 1 步：启动 LMCache 的多进程（MP）server。
# 用前先 `conda activate opencode`。
# 单机单卡冒烟测试用，L1 给 20GB 内存就够了，不用像 disagg 例子那样开 100GB。
set -euo pipefail

lmcache server \
    --l1-size-gb 20 \
    --eviction-policy LRU
