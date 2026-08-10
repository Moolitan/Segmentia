bash scripts/08_lmcache_mp/paper_motivation/3.1/run.sh \
    --skill mcp-builder \
    --overwrite

bash scripts/08_lmcache_mp/paper_motivation/3.1/run.sh \
    --collection Auto-claude-code-research-in-sleep \
    --overwrite

bash scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh \
    --skill internal-comms \
    --max-iterations 20

bash scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh \
    --skill docx \
    --max-iterations 20


bash scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh \
    --skill doc-coauthoring \
    --max-iterations 20

bash scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh \
    --skill frontend-design \
    --max-iterations 20

bash scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh \
    --skill mcp-builder \
    --max-iterations 20

bash scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh \
    --skill idea-discovery \
    --skill research-lit \
    --skill idea-creator \
    --skill novelty-check \
    --skill research-review \
    --skill research-refine-pipeline \
    --max-iterations 20

bash scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh \
    --skill using-superpowers \
    --skill systematic-debugging \
    --max-iterations 20


bash scripts/08_lmcache_mp/paper_motivation/3.1/run_interactive_agent.sh \
    --collection superpowers \
    --max-iterations 20