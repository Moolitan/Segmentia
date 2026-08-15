# CSKCache 实验命令备忘录
#
# 这个文件只用于记录和复制命令，不是可执行的一键脚本；
# 请不要使用 `bash scripts/08_lmcache_mp/paper/run.sh` 执行整个文件。

# 0. 进入仓库并激活实验环境
cd /home/wsh/openhands_code_research
conda activate opencode

# 1. 可选：只检查 Skill 发现、路径和参数，不启动 vLLM
bash scripts/08_lmcache_mp/paper/paper_motivation/3.1/run.sh \
    --skill doc-coauthoring \
    --dry-run

# 2. 将一个 Skill 直接离线保存到 990 PRO 上的 raw-block 容器
# --overwrite 表示重新计算该 Skill。如果上次进程已写入 raw 字节、但未持久化
# LMCache 索引，旧的未发布 pending 会先被保留到 raw/.failed/ 再重建。
bash scripts/08_lmcache_mp/paper/paper_motivation/3.1/run.sh \
    --skill doc-coauthoring \
    --overwrite

# 3. 检查 raw-block 容器、CSKCache Catalog 和构建回执
ls -lh /mnt/990_pro/skill_save_pool/Qwen3-14B/raw/skill_kv.bin
ls -lh /mnt/990_pro/skill_save_pool/Qwen3-14B/raw/catalog.json
ls -lh /mnt/990_pro/skill_save_pool/Qwen3-14B/raw/build.json
python -m json.tool /mnt/990_pro/skill_save_pool/Qwen3-14B/raw/catalog.json

# 4. 运行一个使用新 raw Catalog 的交互式 Agent
# 启动器会先认证 doc-coauthoring 当前 SKILL.md 与离线 token 身份一致，
# 再启动 vLLM。进入交互界面后输入任务，使用 /exit 退出。
bash scripts/08_lmcache_mp/paper/paper_motivation/3.1/run_interactive_agent.sh \
    --skill doc-coauthoring \
    --max-iterations 6

# 5. 运行固定任务的 CSKCache 端到端系统测试
bash scripts/08_lmcache_mp/paper/paper_experiment/cskcache_end_to_end/run.sh
