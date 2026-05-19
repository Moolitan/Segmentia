set -e

python scripts/04_kv_cache_research/kvreused_naturalreferenceblock_root/run_supplement_exp2_gap_scan.py \
    --attention-only \
    --middle-gaps 50,100,200,400,800,1600,2000 \
    --attention-gaps all \
    --attention-layers 5,10,15,25,30 \
    --attention-decode-steps 64


python scripts/04_kv_cache_research/kvreused_naturalreferenceblock_root/run_supplement_exp2_gap_scan_no_query.py \
    --attention-only \
    --middle-gaps 50,100,200,400,800,1600,2000 \
    --attention-gaps all \
    --attention-layers 5,10,15,25,30 \
    --attention-decode-steps 64

python scripts/04_kv_cache_research/kvreused_naturalreferenceblock_root/run_supplement_exp1_multi_prompt.py \
    --attention-only \
    --attention-cases all \
    --attention-layers 5,10,15,25,30 \
    --attention-decode-steps 64

python scripts/04_kv_cache_research/kvreused_naturalreferenceblock_root/run_supplement_exp1_multi_prompt_no_query.py \
    --attention-only \
    --attention-cases all \
    --attention-layers 5,10,15,25,30 \
    --attention-decode-steps 64

python scripts/04_kv_cache_research/kvreused_naturalreferenceblock_root/run_supplement_exp2_a_region_attention.py \
    --attention-only \
    --middle-gaps 50,100,200,400,800,1600,2000 \
    --attention-gaps all \
    --attention-layers 5,10,15,25,30 \
    --attention-decode-steps 64
