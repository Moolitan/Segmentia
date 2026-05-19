#!/usr/bin/env python3
"""Convert a Markdown file to PDF using markdown-it + weasyprint, with CJK support."""

import argparse
import os
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from weasyprint import HTML


def build_html(md_content: str, title: str, figures_dir: str | None = None) -> str:
    """Render Markdown to full HTML document with embedded CSS."""
    md = MarkdownIt("commonmark", {"html": True, "typographer": True})
    md.enable("table")
    md.enable("strikethrough")

    body = md.render(md_content)

    # Resolve relative image paths to absolute file:// URIs if figures_dir given
    if figures_dir:
        abs_dir = Path(figures_dir).expanduser().resolve()

        def _resolve(m):
            src = m.group(1)
            if not src.startswith(("http://", "https://", "file://", "/")):
                abs_path = (abs_dir / src).resolve()
                if abs_path.exists():
                    return f'src="{abs_path.as_uri()}"'
            return m.group(0)

        body = re.sub(r'src="([^"]+)"', _resolve, body)

    css = r"""
    @page {
        size: A4;
        margin: 2cm 2.2cm;
        @bottom-center {
            content: counter(page);
            font-size: 9pt;
            color: #999;
        }
    }
    body {
        font-family: "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                     "PingFang SC", "Microsoft YaHei", "WenQuanYi Micro Hei",
                     "Hiragino Sans GB", sans-serif;
        font-size: 10.5pt;
        line-height: 1.7;
        color: #1a1a1a;
        max-width: 100%;
    }
    h1 {
        font-size: 18pt;
        text-align: center;
        margin-top: 0.5cm;
        margin-bottom: 0.3cm;
        color: #111;
        border-bottom: 2px solid #2563EB;
        padding-bottom: 0.3cm;
    }
    h2 {
        font-size: 14pt;
        color: #1e3a5f;
        margin-top: 1cm;
        border-bottom: 1px solid #ccc;
        padding-bottom: 0.15cm;
    }
    h3 {
        font-size: 12pt;
        color: #2563EB;
        margin-top: 0.6cm;
    }
    h4 {
        font-size: 11pt;
        color: #333;
    }
    p {
        text-align: justify;
        margin: 0.3cm 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.4cm 0;
        font-size: 9.5pt;
    }
    th, td {
        border: 1px solid #ccc;
        padding: 4px 8px;
        text-align: left;
    }
    th {
        background-color: #f0f4ff;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #fafafa;
    }
    code {
        font-family: "JetBrains Mono", "Fira Code", "Source Code Pro",
                     "Noto Sans Mono CJK SC", monospace;
        background-color: #f5f5f5;
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 9pt;
    }
    pre {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px 14px;
        overflow-x: auto;
        font-size: 8.5pt;
        line-height: 1.5;
    }
    pre code {
        background: none;
        padding: 0;
    }
    blockquote {
        border-left: 3px solid #2563EB;
        margin: 0.4cm 0;
        padding: 0.2cm 0.6cm;
        background-color: #f0f7ff;
        color: #333;
    }
    strong {
        color: #111;
    }
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 0.6cm 0;
    }
    img {
        max-width: 100%;
        display: block;
        margin: 0.3cm auto;
    }
    /* Title page subtitle */
    h1 + p {
        text-align: center;
        color: #666;
    }
    """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Markdown → PDF (CJK)")
    parser.add_argument("input", help="Input .md file")
    parser.add_argument("-o", "--output", help="Output .pdf file")
    parser.add_argument("--figures-dir", help="Base dir to resolve relative images")
    parser.add_argument("--title", default="Document")
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.input)[0] + ".pdf"

    with open(args.input, encoding="utf-8") as f:
        md_content = f.read()

    # Extract title from first # heading
    m = re.search(r"^#\s+(.+)", md_content, re.MULTILINE)
    title = m.group(1) if m else args.title

    html = build_html(md_content, title, args.figures_dir)

    print(f"Rendering {args.input} → {args.output} ...")
    # base_url ensures relative resources (img/link/css) resolve reliably in WeasyPrint
    base_url = (
        str(Path(args.figures_dir).expanduser().resolve())
        if args.figures_dir
        else str(Path(args.input).expanduser().resolve().parent)
    )
    HTML(string=html, base_url=base_url).write_pdf(args.output)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"[OK] {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
