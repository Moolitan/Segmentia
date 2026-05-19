"""
扫描 SWE-Bench issue 描述中的高频关键词，辅助设计 Skill KeywordTrigger。

用法：
    python scripts/characterization/scan_issue_keywords.py <output-jsonl> [选项]

参数：
    output-jsonl        评估结果的 output.jsonl 文件路径

选项：
    --top N             显示前 N 个高频词（默认 80）
    --min-len L         词语最短字符数（默认 4）
    --min-count C       最低出现次数（默认 5）
    --stopwords         过滤常见停用词（默认开启）
    --phrases           同时统计 2-gram 短语（默认开启）
    --top-phrases N     显示前 N 个高频短语（默认 40）

输出：
    - 高频单词 Top N（按出现实例数排序，非总词频）
    - 高频 2-gram 短语 Top N
    - 推荐的 Skill 触发词分组（按软件工程场景归类）
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

# 通用英文停用词 + 代码无关词
STOPWORDS = {
    "this", "that", "with", "from", "have", "when", "they", "been",
    "will", "also", "then", "than", "such", "some", "more", "into",
    "which", "would", "could", "should", "there", "their", "about",
    "after", "before", "other", "these", "those", "where", "while",
    "using", "used", "uses", "make", "made", "need", "needs", "like",
    "just", "only", "both", "each", "even", "here", "same", "being",
    "because", "since", "through", "during", "against", "between",
    "over", "under", "above", "below", "very", "well", "what",
    "does", "doing", "done", "call", "calls", "called", "case",
    "code", "file", "line", "lines", "example", "class", "object",
    "value", "values", "name", "names", "data", "item", "items",
    "list", "dict", "string", "number", "current", "given", "added",
    "adds", "note", "show", "shown", "shows", "work", "works",
    "pass", "fails", "fail", "issue", "issues", "problem", "change",
    "changes", "expected", "result", "results", "output", "outputs",
    "input", "inputs", "following", "below", "above", "related",
    "based", "method", "methods", "function", "functions",
}


def extract_issue_text(obj: dict) -> str:
    """从 output.jsonl 的一条记录中提取 issue 文本。"""
    instruction = obj.get("instruction", "")
    # instruction 格式：<uploaded_files>...\n issue_description 内容
    # 尝试提取 <issue_description> 标签内的内容
    match = re.search(r"<issue_description>(.*?)</issue_description>", instruction, re.DOTALL)
    if match:
        return match.group(1)
    # fallback：去掉 <uploaded_files> 块后返回剩余内容
    text = re.sub(r"<uploaded_files>.*?</uploaded_files>", "", instruction, flags=re.DOTALL)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """提取小写字母单词。"""
    return re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_jsonl", help="output.jsonl 文件路径")
    parser.add_argument("--top", type=int, default=80, help="显示前 N 个单词（默认 80）")
    parser.add_argument("--min-len", type=int, default=4, help="词语最短字符数（默认 4）")
    parser.add_argument("--min-count", type=int, default=5, help="最低出现实例数（默认 5）")
    parser.add_argument("--no-stopwords", action="store_true", help="不过滤停用词")
    parser.add_argument("--no-phrases", action="store_true", help="不统计 2-gram 短语")
    parser.add_argument("--top-phrases", type=int, default=40, help="显示前 N 个短语（默认 40）")
    args = parser.parse_args()

    # 读取所有 issue 文本
    issues: list[str] = []
    with open(args.output_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = extract_issue_text(obj)
                if text:
                    issues.append(text)
            except Exception:
                continue

    print(f"共读取 {len(issues)} 个 issue\n")

    # 按实例统计（一个词在同一 issue 中出现多次只算一次）
    word_instance_count: Counter = Counter()
    phrase_instance_count: Counter = Counter()

    for text in issues:
        words = tokenize(text)

        # 过滤
        filtered = [
            w for w in words
            if len(w) >= args.min_len
            and (args.no_stopwords or w not in STOPWORDS)
        ]

        word_instance_count.update(set(filtered))

        if not args.no_phrases:
            bigrams = {f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered) - 1)}
            phrase_instance_count.update(bigrams)

    # 输出高频单词
    print(f"{'=' * 60}")
    print(f"高频单词 Top {args.top}（按出现 issue 数排序）")
    print(f"{'=' * 60}")
    print(f"{'词语':<25} {'出现issue数':>10}  {'占比':>6}")
    print("-" * 50)
    n_issues = len(issues)
    for word, count in word_instance_count.most_common(args.top):
        if count < args.min_count:
            break
        pct = count / n_issues * 100
        print(f"  {word:<23} {count:>10}   {pct:5.1f}%")

    # 输出高频短语
    if not args.no_phrases:
        print(f"\n{'=' * 60}")
        print(f"高频 2-gram 短语 Top {args.top_phrases}")
        print(f"{'=' * 60}")
        print(f"{'短语':<35} {'出现issue数':>10}  {'占比':>6}")
        print("-" * 60)
        for phrase, count in phrase_instance_count.most_common(args.top_phrases):
            if count < args.min_count:
                break
            pct = count / n_issues * 100
            print(f"  {phrase:<33} {count:>10}   {pct:5.1f}%")

    # 推荐的 Skill 触发词场景分类（基于常见 SWE-Bench 主题）
    SCENARIO_KEYWORDS = {
        "测试 / 断言":    ["test", "assert", "pytest", "unittest", "fixture", "mock", "coverage"],
        "类型 / 注解":    ["type", "annotation", "isinstance", "typevar", "optional", "union", "cast", "typed"],
        "异步 / 并发":    ["async", "await", "coroutine", "asyncio", "thread", "lock", "concurrent", "timeout"],
        "错误 / 异常":    ["error", "exception", "raise", "traceback", "stacktrace", "warning", "deprecat"],
        "导入 / 模块":    ["import", "module", "package", "namespace", "circular", "relative"],
        "序列化 / 解析":  ["serialize", "deserialize", "parse", "decode", "encode", "json", "yaml", "pickle"],
        "数据库 / ORM":   ["database", "query", "migration", "transaction", "model", "field", "schema", "queryset"],
        "HTTP / API":     ["request", "response", "endpoint", "status", "header", "middleware", "router", "client"],
        "文件 / 路径":    ["path", "directory", "filesystem", "open", "read", "write", "stream", "encoding"],
        "性能 / 缓存":    ["cache", "memory", "performance", "optimize", "lazy", "batch", "index", "slow"],
    }

    print(f"\n{'=' * 60}")
    print("推荐 Skill 场景分类（出现 issue 数 / 占比）")
    print(f"{'=' * 60}")
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        hits = [(kw, word_instance_count.get(kw, 0)) for kw in keywords]
        hits_sorted = sorted(hits, key=lambda x: -x[1])
        top_hits = [(kw, c) for kw, c in hits_sorted if c >= args.min_count]
        if top_hits:
            kw_str = ", ".join(f"{kw}({c})" for kw, c in top_hits[:5])
            print(f"  {scenario:<18}  {kw_str}")


if __name__ == "__main__":
    main()
