"""
多模态任务序列模板。

定义 TaskSpec / SequenceTemplate 数据类以及所有模板和 theme 配置。
由 run_multimodal_sequence.py 导入使用。
"""

from dataclasses import dataclass, field

FINISH_SUFFIX = " After completing this task, call the finish tool."


@dataclass
class TaskSpec:
    """单个任务规格（去重后的原子任务定义）。"""
    task_id: str                         # 唯一标识符
    message: str                         # 用户消息（可含 {theme} 占位符）
    expected_skills: list[str]           # 预期触发的 Skills
    description: str                     # 人类可读描述


@dataclass
class SequenceTemplate:
    """多轮任务序列模板。"""
    template_id: str                 # e.g. "T4-A"
    description: str                 # 模板描述
    turns: list[TaskSpec] = field(default_factory=list)


# ============================================================================
# 所有 TaskSpec 定义（按 skill 分组，去重）
# ============================================================================

# --- docx: Word 文档创建/编辑 ---

DOCX_THEME_REPORT = TaskSpec(
    task_id="docx_theme_report",
    message="Create a Word document (.docx) with a {theme} report. Include an executive summary, 3 sections with headings, and a conclusion. Save as report.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Create Word report",
)

DOCX_ANALYSIS_SUMMARY = TaskSpec(
    task_id="docx_analysis_summary",
    message="Write a Word document (.docx) summarizing the analysis results with key insights and recommendations. Save as analysis_report.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word summary",
)

DOCX_FINANCIAL_REPORT = TaskSpec(
    task_id="docx_financial_report",
    message="Write a Word document (.docx) as a formal financial report based on the model. Include tables and an executive summary. Save as finance_report.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word financial report",
)

DOCX_PROJECT_PROPOSAL = TaskSpec(
    task_id="docx_project_proposal",
    message="Write a detailed Word document (.docx) as a project proposal expanding on the pitch. Save as proposal.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word proposal",
)

DOCX_SURVEY_REPORT = TaskSpec(
    task_id="docx_survey_report",
    message="Write a Word document (.docx) with the survey analysis report. Include methodology, findings, and recommendations. Save as survey_report.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word survey report",
)

DOCX_MCP_DOCS = TaskSpec(
    task_id="docx_mcp_docs",
    message="Write a Word document (.docx) documenting the MCP server — overview, tool descriptions, example requests/responses, and setup instructions. Save as mcp_docs.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word MCP documentation",
)

DOCX_PERF_AUDIT = TaskSpec(
    task_id="docx_perf_audit",
    message="Write a Word document (.docx) as a performance audit report. Document each optimization made, the before/after patterns, and expected performance improvements. Save as perf_audit.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word performance audit",
)

DOCX_SKILL_GUIDE = TaskSpec(
    task_id="docx_skill_guide",
    message="Write a Word document (.docx) as the skill's user guide — overview, when to use it, example prompts, expected behavior, and troubleshooting. Save as skill_guide.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word skill guide",
)

DOCX_CODE_REVIEW = TaskSpec(
    task_id="docx_code_review",
    message="Write a Word document (.docx) documenting the findings — issues found, root causes, and recommended fixes. Save as code_review.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word code review doc",
)

DOCX_DESIGN_SPEC = TaskSpec(
    task_id="docx_design_spec",
    message="Write a Word document (.docx) as a design specification describing the page layout, color scheme, typography, and responsive behavior. Save as design_spec.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word design spec",
)

DOCX_PERF_REPORT = TaskSpec(
    task_id="docx_perf_report",
    message="Write a Word document (.docx) with a monthly performance report based on the metrics. Save as performance_report.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word performance report",
)

DOCX_TECH_SPEC = TaskSpec(
    task_id="docx_tech_spec",
    message="Write a Word document (.docx) as a technical specification for a {theme} system. Include architecture overview, API design, and data flow. Save as tech_spec.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word tech spec",
)

DOCX_PROJECT_CHARTER = TaskSpec(
    task_id="docx_project_charter",
    message="Write a Word document (.docx) as the project charter based on the kickoff. Include objectives, deliverables, timeline, and governance. Save as charter.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word project charter",
)

DOCX_SYSTEM_ARCH = TaskSpec(
    task_id="docx_system_arch",
    message="Write a Word document (.docx) as the system architecture document covering both the MCP server and the frontend. Include API design, component hierarchy, data flow, and deployment instructions. Save as system_docs.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word system documentation",
)

DOCX_SYSTEM_ARCH_SKILL = TaskSpec(
    task_id="docx_system_arch",
    message="Write a Word document (.docx) as the system architecture document covering both the MCP server and the frontend using 'docx' skill. Include API design, component hierarchy, data flow, and deployment instructions. Save as system_docs.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word system documentation",
)

DOCX_FULLSTACK_DESIGN = TaskSpec(
    task_id="docx_fullstack_design",
    message="Write a Word document (.docx) as the full-stack technical design document. Cover frontend architecture, backend API design, data models, and integration patterns. Save as design_doc.docx" + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Word design document",
)

# --- pptx: PowerPoint 演示文稿 ---

PPTX_KEY_FINDINGS = TaskSpec(
    task_id="pptx_key_findings",
    message="Now create a PowerPoint presentation (.pptx) summarizing the key findings from the report. Make 5 slides with titles and bullet points. Save as summary.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="Create PPT from report",
)

PPTX_BOARD_MEETING = TaskSpec(
    task_id="pptx_board_meeting",
    message="Create a PowerPoint presentation (.pptx) for the board meeting summarizing the financial results. 6 slides. Save as finance_deck.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT board deck",
)

PPTX_PROJECT_PITCH = TaskSpec(
    task_id="pptx_project_pitch",
    message="Create a PowerPoint presentation (.pptx) pitching a {theme} project. Include problem statement, solution, timeline, and budget slides. Save as pitch.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="Create pitch deck",
)

PPTX_SURVEY_HIGHLIGHTS = TaskSpec(
    task_id="pptx_survey_highlights",
    message="Create a PowerPoint presentation (.pptx) with the survey highlights for stakeholders. 5 slides. Save as survey_deck.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT survey deck",
)

PPTX_STANDUP = TaskSpec(
    task_id="pptx_standup",
    message="Create a PowerPoint presentation (.pptx) for the team standup summarizing the code review. 4 slides. Save as standup.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT standup deck",
)

PPTX_DESIGN_REVIEW = TaskSpec(
    task_id="pptx_design_review",
    message="Create a PowerPoint presentation (.pptx) showing the design to stakeholders. Include screenshots concepts and rationale. 5 slides. Save as design_review.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT design review",
)

PPTX_EXEC_REVIEW = TaskSpec(
    task_id="pptx_exec_review",
    message="Create a PowerPoint presentation (.pptx) for the executive review meeting. 6 slides highlighting trends. Save as exec_review.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT executive review",
)

PPTX_ARCH_REVIEW = TaskSpec(
    task_id="pptx_arch_review",
    message="Create a PowerPoint presentation (.pptx) for the architecture review. Include system diagrams described in text and key design decisions. 5 slides. Save as arch_review.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT architecture review",
)

PPTX_KICKOFF = TaskSpec(
    task_id="pptx_kickoff",
    message="Create a PowerPoint presentation (.pptx) for a {theme} project kickoff meeting. Include goals, scope, milestones, team roles, and risks. 7 slides. Save as kickoff.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT kickoff deck",
)

PPTX_SKILL_PITCH = TaskSpec(
    task_id="pptx_skill_pitch",
    message="Create a PowerPoint presentation (.pptx) pitching the new skill to stakeholders. Include problem statement, competitive analysis, demo highlights, and adoption plan. 6 slides. Save as skill_pitch.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT skill pitch",
)

PPTX_DEMO_DAY = TaskSpec(
    task_id="pptx_demo_day",
    message="Create a PowerPoint presentation (.pptx) for the product demo day. Show the architecture, key features, performance improvements, and roadmap. 7 slides. Save as demo_day.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT demo day deck",
)

PPTX_DEMO_DAY_SKILL = TaskSpec(
    task_id="pptx_demo_day",
    message="Create a PowerPoint presentation (.pptx) for the product demo day using 'pptx' skill. Show the architecture, key features, performance improvements, and roadmap. 7 slides. Save as demo_day.pptx" + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="PPT demo day deck",
)

# --- xlsx: Excel 电子表格 ---

XLSX_REPORT_DATA = TaskSpec(
    task_id="xlsx_report_data",
    message="Create an Excel spreadsheet (.xlsx) with the numerical data referenced in the report. Include formulas for totals and averages. Save as data.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Create Excel data",
)

XLSX_DATA_ANALYSIS = TaskSpec(
    task_id="xlsx_data_analysis",
    message="Create an Excel spreadsheet (.xlsx) analyzing this data. Add summary statistics, a pivot-style summary, and conditional formatting. Save as analysis.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel analysis",
)

XLSX_FINANCIAL_MODEL = TaskSpec(
    task_id="xlsx_financial_model",
    message="Create an Excel spreadsheet (.xlsx) with a {theme} financial model. Include revenue, costs, and profit calculations with formulas. Save as finance.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Create financial model",
)

XLSX_PROJECT_BUDGET = TaskSpec(
    task_id="xlsx_project_budget",
    message="Create an Excel spreadsheet (.xlsx) with the project budget breakdown and timeline. Include formulas for totals. Save as budget.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel budget",
)

XLSX_MCP_INVENTORY = TaskSpec(
    task_id="xlsx_mcp_inventory",
    message="Create an Excel spreadsheet (.xlsx) with an MCP tool inventory: tool name, description, parameters, return type, and example usage. Include conditional formatting for required vs optional params. Save as mcp_inventory.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel tool inventory",
)

XLSX_ISSUE_TRACKER = TaskSpec(
    task_id="xlsx_issue_tracker",
    message="Create an Excel spreadsheet (.xlsx) tracking all issues with columns: ID, Module, Severity, Status, Assignee. Include conditional formatting. Save as issue_tracker.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel issue tracker",
)

XLSX_CONTENT_INVENTORY = TaskSpec(
    task_id="xlsx_content_inventory",
    message="Create an Excel spreadsheet (.xlsx) with a content inventory: list all sections, headings, CTAs, and images. Save as content_inventory.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel content inventory",
)

XLSX_METRICS_DATA = TaskSpec(
    task_id="xlsx_metrics_data",
    message="Create an Excel spreadsheet (.xlsx) with sample {theme} metrics data — 12 months of KPIs with formulas. Save as metrics.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Create metrics spreadsheet",
)

XLSX_API_INVENTORY = TaskSpec(
    task_id="xlsx_api_inventory",
    message="Create an Excel spreadsheet (.xlsx) with an API endpoint inventory — endpoint, method, auth required, rate limit. Save as api_inventory.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel API inventory",
)

XLSX_PROJECT_PLAN = TaskSpec(
    task_id="xlsx_project_plan",
    message="Create an Excel spreadsheet (.xlsx) with the project plan — tasks, owners, start/end dates, dependencies, and status. Save as project_plan.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel project plan",
)

XLSX_API_REFERENCE = TaskSpec(
    task_id="xlsx_api_reference",
    message="Create an Excel spreadsheet (.xlsx) with the complete API reference — endpoint, method, parameters, response schema, auth requirement, and rate limits. Include auto-filters. Save as api_reference.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel API reference",
)

XLSX_API_REFERENCE_SKILL = TaskSpec(
    task_id="xlsx_api_reference",
    message="Create an Excel spreadsheet (.xlsx) with the complete API reference using 'xlsx' skill— endpoint, method, parameters, response schema, auth requirement, and rate limits. Include auto-filters. Save as api_reference.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel API reference",
)

XLSX_SKILL_COMPARISON = TaskSpec(
    task_id="xlsx_skill_comparison",
    message="Create an Excel spreadsheet (.xlsx) with a skill comparison matrix: feature name, our skill vs competitors, rating, and notes. Include conditional formatting for strengths/weaknesses. Save as skill_comparison.xlsx" + FINISH_SUFFIX,
    expected_skills=["xlsx"],
    description="Excel skill comparison",
)

# --- jpeg: JPEG 图片导出 ---

IMAGE_EXEC_SUMMARY = TaskSpec(
    task_id="jpeg_exec_summary",
    message="Convert exec_review.pptx to JPEG images for distribution. Save as exec_review-01.jpg, exec_review-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="Export PPTX to JPEG",
)

IMAGE_ANALYSIS_REPORT = TaskSpec(
    task_id="jpeg_analysis_report",
    message="Convert analysis_report.docx to JPEG images. Save as analysis_report-01.jpg, analysis_report-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_FINANCIAL_REPORT = TaskSpec(
    task_id="jpeg_financial_report",
    message="Export the financial report as JPEG images for archival. Convert finance_report.docx and save as finance_report-01.jpg, finance_report-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_PROPOSAL = TaskSpec(
    task_id="jpeg_proposal",
    message="Convert proposal.docx to JPEG images. Save as proposal-01.jpg, proposal-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_SURVEY_REPORT = TaskSpec(
    task_id="jpeg_survey_report",
    message="Export survey_report.docx as JPEG images. Save as survey_report-01.jpg, survey_report-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_MCP_DOCS = TaskSpec(
    task_id="jpeg_mcp_docs",
    message="Convert mcp_docs.docx to JPEG images. Save as mcp_docs-01.jpg, mcp_docs-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_PERF_AUDIT = TaskSpec(
    task_id="jpeg_perf_audit",
    message="Convert perf_audit.docx to JPEG images. Save as perf_audit-01.jpg, perf_audit-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_SKILL_GUIDE = TaskSpec(
    task_id="jpeg_skill_guide",
    message="Convert skill_guide.docx to JPEG images for distribution. Save as skill_guide-01.jpg, skill_guide-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_FINAL_REPORT = TaskSpec(
    task_id="jpeg_final_report",
    message="Convert standup.pptx to JPEG images. Save as standup-01.jpg, standup-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="Export PPTX to JPEG",
)

IMAGE_DESIGN_SPEC = TaskSpec(
    task_id="jpeg_design_spec",
    message="Convert design_spec.docx to JPEG images. Save as design_spec-01.jpg, design_spec-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_PERF_REPORT = TaskSpec(
    task_id="jpeg_perf_report",
    message="Export performance_report.docx as JPEG images. Save as performance_report-01.jpg, performance_report-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_TECH_SPEC = TaskSpec(
    task_id="jpeg_tech_spec",
    message="Convert tech_spec.docx to JPEG images. Save as tech_spec-01.jpg, tech_spec-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_CHARTER = TaskSpec(
    task_id="jpeg_charter",
    message="Convert charter.docx to JPEG images for distribution. Save as charter-01.jpg, charter-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_SYSTEM_DOCS = TaskSpec(
    task_id="jpeg_system_docs",
    message="Convert system_docs.docx to JPEG images. Save as system_docs-01.jpg, system_docs-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_SYSTEM_DOCS_SKILL = TaskSpec(
    task_id="jpeg_system_docs",
    message="Convert system_docs.docx to JPEG images using 'docx' skill. Save as system_docs-01.jpg, system_docs-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

IMAGE_SKILL_OVERVIEW = TaskSpec(
    task_id="jpeg_skill_overview",
    message="Convert skill_pitch.pptx to JPEG images for distribution. Save as skill_pitch-01.jpg, skill_pitch-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["pptx"],
    description="Export PPTX to JPEG",
)

IMAGE_DESIGN_DOC = TaskSpec(
    task_id="jpeg_design_doc",
    message="Convert design_doc.docx to JPEG images. Save as design_doc-01.jpg, design_doc-02.jpg, etc." + FINISH_SUFFIX,
    expected_skills=["docx"],
    description="Export DOCX to JPEG",
)

# --- file-reading: 文件读取/分析 ---

READ_CSV_DATA = TaskSpec(
    task_id="read_csv_data",
    message="I have a CSV file with {theme} data. Read it and describe what's in it — how many rows, columns, and key patterns you see." + FINISH_SUFFIX,
    expected_skills=["file-reading"],
    description="Read input file",
)

READ_SURVEY_RESULTS = TaskSpec(
    task_id="read_survey_results",
    message="Read a survey results file about {theme}. Identify the key questions, response patterns, and sample size." + FINISH_SUFFIX,
    expected_skills=["file-reading"],
    description="Read survey data",
)

READ_SOURCE_CODE = TaskSpec(
    task_id="read_source_code",
    message="Read the source code file for a {theme} module and identify potential issues or improvements." + FINISH_SUFFIX,
    expected_skills=["file-reading"],
    description="Read source code",
)

READ_HTML_ANALYSIS = TaskSpec(
    task_id="read_html_analysis",
    message="Read the generated HTML file and analyze its structure — count elements, identify sections, check accessibility basics." + FINISH_SUFFIX,
    expected_skills=["file-reading"],
    description="Read and analyze HTML",
)

READ_METRICS = TaskSpec(
    task_id="read_metrics",
    message="Read the metrics spreadsheet and summarize the key trends and anomalies you find." + FINISH_SUFFIX,
    expected_skills=["file-reading"],
    description="Read and analyze metrics",
)

READ_TECH_SPEC = TaskSpec(
    task_id="read_tech_spec",
    message="Read the tech spec document and extract all API endpoints, their methods, and parameters." + FINISH_SUFFIX,
    expected_skills=["file-reading"],
    description="Read and extract from spec",
)

READ_PROJECT_PLAN = TaskSpec(
    task_id="read_project_plan",
    message="Read the project plan spreadsheet and summarize the critical path and any scheduling risks." + FINISH_SUFFIX,
    expected_skills=["file-reading"],
    description="Read and analyze plan",
)

# --- frontend-design: 前端界面设计 ---

FRONTEND_QUALITY_DASHBOARD = TaskSpec(
    task_id="frontend_quality_dashboard",
    message="Create a frontend dashboard (single HTML file) that displays code quality metrics — issues by severity, fix status, module breakdown. Make it visually polished. Save as dashboard.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Frontend metrics dashboard",
)

FRONTEND_LANDING_PAGE = TaskSpec(
    task_id="frontend_landing_page",
    message="Create a landing page (single HTML file) for a {theme} product. Make it visually striking with modern design. Save as landing.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Frontend landing page",
)

FRONTEND_METRICS_DASHBOARD = TaskSpec(
    task_id="frontend_metrics_dashboard",
    message="Create an interactive dashboard (single HTML file) visualizing the metrics with charts. Make it professional. Save as metrics_dashboard.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Frontend metrics dashboard",
)

FRONTEND_API_DOCS = TaskSpec(
    task_id="frontend_api_docs",
    message="Create a single-page web app (HTML file) that serves as interactive API documentation for the system. Include endpoint descriptions and example requests. Save as api_docs.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Frontend API docs",
)

FRONTEND_KANBAN_BOARD = TaskSpec(
    task_id="frontend_kanban_board",
    message="Create a project status board (single HTML file) showing tasks by status (To Do, In Progress, Done) in a Kanban layout. Save as status_board.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Frontend Kanban board",
)

FRONTEND_REACT_PAGE = TaskSpec(
    task_id="frontend_react_page",
    message="Create a React/Next.js page (single HTML file with inline React via CDN) for a {theme} application. Include multiple components with data fetching, state management, and a polished UI. Save as app.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Create React page",
)

FRONTEND_ADMIN_DASHBOARD = TaskSpec(
    task_id="frontend_admin_dashboard",
    message="Create a React frontend (single HTML file) that serves as an admin dashboard for the {theme} platform. Include data tables, forms, and interactive charts. Make it visually polished. Save as admin_dashboard.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Frontend admin dashboard",
)

FRONTEND_SKILL_DEMO = TaskSpec(
    task_id="frontend_skill_demo",
    message="Build a demo page (single HTML file) showcasing what the {theme} skill can do. Include interactive examples, before/after comparisons, and a polished UI. Save as skill_demo.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Frontend skill demo",
)

FRONTEND_REACT_APP = TaskSpec(
    task_id="frontend_react_app",
    message="Create a Next.js-style React application (single HTML file) for a {theme} product. Include routing simulation, multiple views, responsive design, and a distinctive visual identity. Save as app.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Create React application",
)

FRONTEND_REACT_APP_SKILL = TaskSpec(
    task_id="frontend_react_app",
    message="Create a Next.js-style React application (single HTML file) for a {theme} product using 'frontend-design' skill. Include routing simulation, multiple views, responsive design, and a distinctive visual identity. Save as app.html" + FINISH_SUFFIX,
    expected_skills=["frontend-design"],
    description="Create React application",
)

# --- mcp-builder: MCP 服务器构建 ---

MCP_SERVICE_SERVER = TaskSpec(
    task_id="mcp_service_server",
    message="Build an MCP (Model Context Protocol) server in Python using FastMCP that exposes tools for a {theme} service. Include at least 3 tools with proper descriptions and error handling. Save the server code as mcp_server.py" + FINISH_SUFFIX,
    expected_skills=["mcp-builder"],
    description="Build MCP server",
)

MCP_PLATFORM_SERVER = TaskSpec(
    task_id="mcp_platform_server",
    message="Build an MCP server in Python using FastMCP for a {theme} platform. Include CRUD tools, search, and aggregation endpoints with proper input validation. Save as mcp_server.py" + FINISH_SUFFIX,
    expected_skills=["mcp-builder"],
    description="Build MCP server",
)

MCP_BACKEND_API = TaskSpec(
    task_id="mcp_backend_api",
    message="Build an MCP server as the backend API for the {theme} product. Implement data management tools, user operations, and analytics endpoints using FastMCP in Python. Save as backend_mcp.py" + FINISH_SUFFIX,
    expected_skills=["mcp-builder"],
    description="Build MCP backend",
)

MCP_BACKEND_API_SKILL = TaskSpec(
    task_id="mcp_backend_api",
    message="Build an MCP server as the backend API for the {theme} product using 'mcp-builder' skill. Implement data management tools, user operations, and analytics endpoints using FastMCP in Python. Save as backend_mcp.py" + FINISH_SUFFIX,
    expected_skills=["mcp-builder"],
    description="Build MCP backend",
)

# --- vercel-react-best-practices: React/Next.js 性能优化 ---

VERCEL_OPTIMIZE_APP = TaskSpec(
    task_id="vercel_optimize_app",
    message="Review the React application code and optimize it following Vercel's React best practices. Fix any performance anti-patterns — eliminate waterfalls, optimize bundle size, improve server-side rendering patterns, and add proper memoization. Save the optimized version as app_optimized.html" + FINISH_SUFFIX,
    expected_skills=["vercel-react-best-practices"],
    description="Optimize with Vercel best practices",
)

VERCEL_OPTIMIZE_DASHBOARD = TaskSpec(
    task_id="vercel_optimize_dashboard",
    message="Review and optimize the React dashboard following Vercel's best practices. Eliminate render waterfalls, add proper code splitting boundaries, optimize re-renders with memoization, and improve data fetching patterns. Save the optimized version as admin_dashboard_optimized.html" + FINISH_SUFFIX,
    expected_skills=["vercel-react-best-practices"],
    description="Optimize React dashboard",
)

VERCEL_PERF_REVIEW = TaskSpec(
    task_id="vercel_perf_review",
    message="Perform a comprehensive React performance review following Vercel engineering best practices. Optimize async data patterns, reduce bundle size, add proper suspense boundaries, fix hydration issues, and implement efficient caching. Save as app_optimized.html" + FINISH_SUFFIX,
    expected_skills=["vercel-react-best-practices"],
    description="Vercel performance optimization",
)

VERCEL_PERF_REVIEW_SKILL = TaskSpec(
    task_id="vercel_perf_review",
    message="Perform a comprehensive React performance review following Vercel engineering best practices using 'vercel-react-best-practices' skill. Optimize async data patterns, reduce bundle size, add proper suspense boundaries, fix hydration issues, and implement efficient caching. Save as app_optimized.html" + FINISH_SUFFIX,
    expected_skills=["vercel-react-best-practices"],
    description="Vercel performance optimization",
)

# --- skill-creator: 技能创建/优化 ---

SKILL_CREATE_NEW = TaskSpec(
    task_id="skill_create_new",
    message="Based on the gaps identified, create a new agent skill for {theme}. Write the SKILL.md file with a clear description, trigger conditions, and step-by-step instructions. Save it in a new directory called {theme}_skill/SKILL.md" + FINISH_SUFFIX,
    expected_skills=["skill-creator"],
    description="Create new skill",
)

SKILL_CREATE_COMPREHENSIVE = TaskSpec(
    task_id="skill_create_comprehensive",
    message="Create a comprehensive new agent skill for {theme}. Write the SKILL.md with detailed instructions, examples, edge cases, and evaluation criteria. Also create test prompts for benchmarking. Save in {theme}_skill/ directory" + FINISH_SUFFIX,
    expected_skills=["skill-creator"],
    description="Create comprehensive skill",
)

# --- find-skills: 技能发现/搜索 ---

FIND_SKILLS_GAPS = TaskSpec(
    task_id="find_skills_gaps",
    message="Search for existing agent skills related to {theme}. List what skills are available, what they do, and identify gaps where a new skill would be valuable. Summarize your findings." + FINISH_SUFFIX,
    expected_skills=["find-skills"],
    description="Discover existing skills",
)

FIND_SKILLS_ECOSYSTEM = TaskSpec(
    task_id="find_skills_ecosystem",
    message="Search for existing agent skills related to {theme}. Analyze what's available in the ecosystem, compare their features, and identify opportunities for a better or complementary skill." + FINISH_SUFFIX,
    expected_skills=["find-skills"],
    description="Research skill ecosystem",
)

FIND_SKILLS_ECOSYSTEM_SKILL = TaskSpec(
    task_id="find_skills_ecosystem",
    message="Search for existing agent skills related to {theme} using 'find-skills' skill. Analyze what's available in the ecosystem, compare their features, and identify opportunities for a better or complementary skill." + FINISH_SUFFIX,
    expected_skills=["find-skills"],
    description="Research skill ecosystem",
)

# ============================================================================
# 序列模板定义（引用 TaskSpec 实例）
# ============================================================================

T2_A = SequenceTemplate(
    # 快速两轮：先建页面，再做 Vercel 最佳实践优化
    template_id="T2-A",
    description="Quick React: Frontend → Vercel",
    turns=[FRONTEND_REACT_PAGE, VERCEL_OPTIMIZE_APP],
)

T2_B = SequenceTemplate(
    # 快速两轮：先建 Excel 模型，再写 Word 报告
    template_id="T2-B",
    description="Quick reporting: Excel → Word",
    turns=[XLSX_FINANCIAL_MODEL, DOCX_FINANCIAL_REPORT],
)


T4_A = SequenceTemplate(
    # 有数据才能分析，有分析才能写报告，有报告才能归档
    template_id="T4-A",
    description="Data analysis: Read → Excel → Word → JPEG",
    turns=[READ_CSV_DATA, XLSX_DATA_ANALYSIS, DOCX_ANALYSIS_SUMMARY, IMAGE_ANALYSIS_REPORT],
)

T4_B = SequenceTemplate(
    # 先建财务模型，写正式报告，再向董事会汇报
    template_id="T4-B",

    description="Financial reporting: Excel → Word → PPT → JPEG",
    turns=[XLSX_FINANCIAL_MODEL, DOCX_FINANCIAL_REPORT, PPTX_BOARD_MEETING, IMAGE_FINANCIAL_REPORT],
)

T4_C = SequenceTemplate(
    # 先看问卷数据，写分析报告，再做stakeholder演示
    template_id="T4-C",

    description="Survey analysis: Read → Word → PPT → JPEG",
    turns=[READ_SURVEY_RESULTS, DOCX_SURVEY_REPORT, PPTX_SURVEY_HIGHLIGHTS, IMAGE_SURVEY_REPORT],
)

T4_D = SequenceTemplate(
    # 先pitch概念，再写详细方案，最后落实预算
    template_id="T4-D",

    description="Project proposal: PPT → Word → Excel → JPEG",
    turns=[PPTX_PROJECT_PITCH, DOCX_PROJECT_PROPOSAL, XLSX_PROJECT_BUDGET, IMAGE_PROPOSAL],
)

T4_E = SequenceTemplate(
    # 先写代码，再写文档和接口清单
    template_id="T4-E",

    description="MCP server development: MCP → Word → Excel → JPEG",
    turns=[MCP_SERVICE_SERVER, DOCX_MCP_DOCS, XLSX_MCP_INVENTORY, IMAGE_MCP_DOCS],
)

T4_F = SequenceTemplate(
    # 先建，再优化，最后出审计报告
    template_id="T4-F",

    description="React optimization: Frontend → Vercel → Word → JPEG",
    turns=[FRONTEND_REACT_PAGE, VERCEL_OPTIMIZE_APP, DOCX_PERF_AUDIT, IMAGE_PERF_AUDIT],
)

T4_G = SequenceTemplate(
    # 先调研，再创建，最后写使用手册
    template_id="T4-G",

    description="Skill development: Find → Create → Word → JPEG",
    turns=[FIND_SKILLS_GAPS, SKILL_CREATE_NEW, DOCX_SKILL_GUIDE, IMAGE_SKILL_GUIDE],
)

T4_H = SequenceTemplate(
    # 先有数据，再可视化，最后出报告
    template_id="T4-H",

    description="Data visualization: Excel → Frontend → Word → JPEG",
    turns=[XLSX_METRICS_DATA, FRONTEND_METRICS_DASHBOARD, DOCX_PERF_REPORT, IMAGE_PERF_REPORT],
)

# --- 6-turn 序列 ---
# 设计原则：完整工作流闭环，每步自然承接上步产出

T6_A = SequenceTemplate(
    # 代码审查：读代码→记录问题→可视化→写报告→汇报
    template_id="T6-A",

    description="Code review: Read → Excel → Frontend → Word → PPT → JPEG",
    turns=[
        READ_SOURCE_CODE, XLSX_ISSUE_TRACKER, FRONTEND_QUALITY_DASHBOARD,
        DOCX_CODE_REVIEW, PPTX_STANDUP, IMAGE_FINAL_REPORT,
    ],
)

T6_B = SequenceTemplate(
    # 全栈开发：前端→优化→后端→接口文档→产品演示
    template_id="T6-B",

    description="Full-stack product: Frontend → Vercel → MCP → Excel → PPT → JPEG",
    turns=[
        FRONTEND_REACT_APP, VERCEL_PERF_REVIEW, MCP_BACKEND_API,
        XLSX_API_REFERENCE, PPTX_DEMO_DAY, IMAGE_DESIGN_DOC,
    ],
)

T6_C = SequenceTemplate(
    # 项目管理：启动→立项→排期→审查→状态看板
    template_id="T6-C",

    description="Project management: PPT → Word → Excel → Read → Frontend → JPEG",
    turns=[
        PPTX_KICKOFF, DOCX_PROJECT_CHARTER, XLSX_PROJECT_PLAN,
        READ_PROJECT_PLAN, FRONTEND_KANBAN_BOARD, IMAGE_CHARTER,
    ],
)

T6_D = SequenceTemplate(
    # 系统文档：读现有文档→写规格→整理接口→交互文档→评审
    template_id="T6-D",

    description="System documentation: Read → Word → Excel → Frontend → PPT → JPEG",
    turns=[
        READ_TECH_SPEC, DOCX_TECH_SPEC, XLSX_API_INVENTORY,
        FRONTEND_API_DOCS, PPTX_ARCH_REVIEW, IMAGE_TECH_SPEC,
    ],
)

T6_E = SequenceTemplate(
    # MCP平台：建服务→建界面→优化→写文档→汇报
    template_id="T6-E",

    description="MCP platform: MCP → Frontend → Vercel → Word → PPT → JPEG",
    turns=[
        MCP_PLATFORM_SERVER, FRONTEND_ADMIN_DASHBOARD, VERCEL_OPTIMIZE_DASHBOARD,
        DOCX_SYSTEM_ARCH, PPTX_KEY_FINDINGS, IMAGE_SYSTEM_DOCS,
    ],
)

T6_F = SequenceTemplate(
    # 技能生态：调研→开发→做Demo→文档→向stakeholder推介
    template_id="T6-F",

    description="Skill ecosystem: Find → Create → Frontend → Word → PPT → JPEG",
    turns=[
        FIND_SKILLS_ECOSYSTEM, SKILL_CREATE_COMPREHENSIVE, FRONTEND_SKILL_DEMO,
        DOCX_THEME_REPORT, PPTX_SKILL_PITCH, IMAGE_SKILL_OVERVIEW,
    ],
)

T6_G = SequenceTemplate(
    # 数据驱动设计：造数据→分析→建页面→写设计规格→评审
    template_id="T6-G",

    description="Data-driven design: Excel → Read → Frontend → Word → PPT → JPEG",
    turns=[
        XLSX_REPORT_DATA, READ_METRICS, FRONTEND_LANDING_PAGE,
        DOCX_DESIGN_SPEC, PPTX_DESIGN_REVIEW, IMAGE_DESIGN_SPEC,
    ],
)

T6_H = SequenceTemplate(
    # 竞品分析：分析页面→盘点内容→写设计方案→竞品对比→汇报高层
    template_id="T6-H",

    description="Competitive analysis: Read → Excel → Word → Excel → PPT → JPEG",
    turns=[
        READ_HTML_ANALYSIS, XLSX_CONTENT_INVENTORY, DOCX_FULLSTACK_DESIGN,
        XLSX_SKILL_COMPARISON, PPTX_EXEC_REVIEW, IMAGE_EXEC_SUMMARY,
    ],
)

# --- 8-turn 序列 ---
# 设计原则：完整端到端工作流，8步自然串联，每步产出下步输入

T8_A_SKILL = SequenceTemplate(
    # 全栈SaaS产品开发：调研→建后端→建前端→优化→API文档→系统文档→演示→归档
    template_id="T8-A-SKILL",
    description="Full-stack SaaS: Find → MCP → Frontend → Vercel → Excel → Word → PPT → JPEG",
    turns=[
        FIND_SKILLS_ECOSYSTEM_SKILL, MCP_BACKEND_API_SKILL, FRONTEND_REACT_APP_SKILL,
        VERCEL_PERF_REVIEW_SKILL, XLSX_API_REFERENCE_SKILL, DOCX_SYSTEM_ARCH_SKILL,
        PPTX_DEMO_DAY_SKILL, IMAGE_SYSTEM_DOCS_SKILL,
    ],
)

T8_A = SequenceTemplate(
    # 全栈SaaS产品开发：调研→建后端→建前端→优化→API文档→系统文档→演示→归档
    template_id="T8-A",
    description="Full-stack SaaS: Find → MCP → Frontend → Vercel → Excel → Word → PPT → JPEG",
    turns=[
        FIND_SKILLS_ECOSYSTEM, MCP_BACKEND_API, FRONTEND_REACT_APP,
        VERCEL_PERF_REVIEW, XLSX_API_REFERENCE, DOCX_SYSTEM_ARCH,
        PPTX_DEMO_DAY, IMAGE_SYSTEM_DOCS,
    ],
)

T8_B = SequenceTemplate(
    # 企业财务分析：读数据→分析→建模→可视化→优化→写报告→汇报→归档
    template_id="T8-B",
    description="Financial analytics: Read → Excel → Excel → Frontend → Vercel → Word → PPT → JPEG",
    turns=[
        READ_CSV_DATA, XLSX_DATA_ANALYSIS, XLSX_FINANCIAL_MODEL,
        FRONTEND_METRICS_DASHBOARD, VERCEL_OPTIMIZE_APP, DOCX_FINANCIAL_REPORT,
        PPTX_BOARD_MEETING, IMAGE_FINANCIAL_REPORT,
    ],
)

T8_C = SequenceTemplate(
    # 项目管理全生命周期：启动→立项→排期→审查→预算→看板→高层汇报→归档
    template_id="T8-C",
    description="Project lifecycle: PPT → Word → Excel → Read → Excel → Frontend → PPT → JPEG",
    turns=[
        PPTX_KICKOFF, DOCX_PROJECT_CHARTER, XLSX_PROJECT_PLAN,
        READ_PROJECT_PLAN, XLSX_PROJECT_BUDGET, FRONTEND_KANBAN_BOARD,
        PPTX_EXEC_REVIEW, IMAGE_CHARTER,
    ],
)

T8_D = SequenceTemplate(
    # 技能开发全流程：调研→创建→做Demo→优化→竞品对比→写手册→推介→归档
    template_id="T8-D",
    description="Skill development: Find → Create → Frontend → Vercel → Excel → Word → PPT → JPEG",
    turns=[
        FIND_SKILLS_GAPS, SKILL_CREATE_COMPREHENSIVE, FRONTEND_SKILL_DEMO,
        VERCEL_OPTIMIZE_APP, XLSX_SKILL_COMPARISON, DOCX_SKILL_GUIDE,
        PPTX_SKILL_PITCH, IMAGE_SKILL_OVERVIEW,
    ],
)

# --- 10-turn 序列 ---
# 设计原则：覆盖全部技能类型的完整端到端闭环，10步形成从调研到交付的完整流水线

T10_A = SequenceTemplate(
    # 企业平台全周期：调研→读规格→建后端→建界面→优化→API清单→造技能→写文档→演示→归档
    template_id="T10-A",
    description="Enterprise platform: Find → Read → MCP → Frontend → Vercel → Excel → Skill → Word → PPT → JPEG",
    turns=[
        FIND_SKILLS_ECOSYSTEM, READ_TECH_SPEC, MCP_PLATFORM_SERVER,
        FRONTEND_ADMIN_DASHBOARD, VERCEL_OPTIMIZE_DASHBOARD, XLSX_API_REFERENCE,
        SKILL_CREATE_NEW, DOCX_SYSTEM_ARCH, PPTX_DEMO_DAY, IMAGE_SYSTEM_DOCS,
    ],
)

T10_B = SequenceTemplate(
    # 数据智能平台：调研→读数据→分析→建数据API→可视化→优化→造技能→写报告→汇报→归档
    template_id="T10-B",
    description="Data intelligence: Find → Read → Excel → MCP → Frontend → Vercel → Skill → Word → PPT → JPEG",
    turns=[
        FIND_SKILLS_GAPS, READ_CSV_DATA, XLSX_DATA_ANALYSIS,
        MCP_SERVICE_SERVER, FRONTEND_METRICS_DASHBOARD, VERCEL_OPTIMIZE_APP,
        SKILL_CREATE_NEW, DOCX_ANALYSIS_SUMMARY, PPTX_KEY_FINDINGS, IMAGE_ANALYSIS_REPORT,
    ],
)

T10_C = SequenceTemplate(
    # 技能生态全建设：调研→开发技能→建后端→做Demo→优化→审查→竞品→写手册→推介→归档
    template_id="T10-C",
    description="Skill ecosystem: Find → Create → MCP → Frontend → Vercel → Read → Excel → Word → PPT → JPEG",
    turns=[
        FIND_SKILLS_ECOSYSTEM, SKILL_CREATE_COMPREHENSIVE, MCP_SERVICE_SERVER,
        FRONTEND_SKILL_DEMO, VERCEL_OPTIMIZE_APP, READ_HTML_ANALYSIS,
        XLSX_SKILL_COMPARISON, DOCX_SKILL_GUIDE, PPTX_SKILL_PITCH, IMAGE_SKILL_OVERVIEW,
    ],
)

T10_D = SequenceTemplate(
    # 技术文档平台：读规格→调研→建文档API→接口清单→交互文档→优化→造技能→写规格→评审→归档
    template_id="T10-D",
    description="Documentation platform: Read → Find → MCP → Excel → Frontend → Vercel → Skill → Word → PPT → JPEG",
    turns=[
        READ_TECH_SPEC, FIND_SKILLS_GAPS, MCP_SERVICE_SERVER,
        XLSX_API_INVENTORY, FRONTEND_API_DOCS, VERCEL_OPTIMIZE_APP,
        SKILL_CREATE_NEW, DOCX_TECH_SPEC, PPTX_ARCH_REVIEW, IMAGE_TECH_SPEC,
    ],
)


# ============================================================================
# 所有模板和 Theme 配置
# ============================================================================

ALL_TEMPLATES: list[SequenceTemplate] = [
    T2_A, T2_B,
    T4_A, T4_B, T4_C, T4_D, T4_E, T4_F, T4_G, T4_H,
    T6_A, T6_B, T6_C, T6_D, T6_E, T6_F, T6_G, T6_H,
    T8_A, T8_A_SKILL, T8_B, T8_C, T8_D,
    T10_A, T10_B, T10_C, T10_D,
]

THEMES: dict[str, list[str]] = {
    "T2-A": ["social_feed", "admin_panel", "ecommerce_store"],
    "T2-B": ["annual_review", "startup_budget", "department_forecast"],
    # 4-turn
    "T4-A": ["employee_survey", "website_traffic", "inventory_audit"],
    "T4-B": ["startup_budget", "department_forecast", "annual_review"],
    "T4-C": ["customer_feedback", "employee_engagement", "market_research"],
    "T4-D": ["mobile_app", "data_migration", "cloud_infrastructure"],
    "T4-E": ["weather_api", "github_integration", "database_connector"],
    "T4-F": ["ecommerce_store", "admin_panel", "social_feed"],
    "T4-G": ["code_review", "data_visualization", "test_automation"],
    "T4-H": ["quarterly_sales", "product_launch", "customer_satisfaction"],
    # 6-turn
    "T6-A": ["authentication_module", "payment_processing", "data_pipeline"],
    "T6-B": ["fitness-tracker", "recipe_sharing", "task_manager"],
    "T6-C": ["platform_rewrite", "security_audit", "ml_pipeline"],
    "T6-D": ["notification_service", "search_engine", "recommendation_system"],
    "T6-E": ["project_management", "crm-system", "analytics_platform"],
    "T6-F": ["devops-automation", "api_testing", "documentation_generator"],
    "T6-G": ["sales_performance", "server_monitoring", "marketing_campaign"],
    "T6-H": ["inventory_management", "user_dashboard", "content_management"],
    # 8-turn
    "T8-A": ["task_manager", "recipe_sharing", "fitness-tracker"],
    "T8-A-SKILL": ["task_manager", "recipe_sharing", "fitness-tracker"],
    "T8-B": ["quarterly_sales", "annual_review", "department_forecast"],
    "T8-C": ["platform_rewrite", "security_audit", "ml_pipeline"],
    "T8-D": ["code_review", "data_visualization", "test_automation"],
    # 10-turn
    "T10-A": ["analytics_platform", "crm-system", "project_management"],
    "T10-B": ["customer_feedback", "employee_engagement", "market_research"],
    "T10-C": ["devops-automation", "api_testing", "documentation_generator"],
    "T10-D": ["notification_service", "search_engine", "recommendation_system"],
}


def get_templates(template_id: str | None = None) -> list[SequenceTemplate]:
    """根据 template_id 过滤模板。"""
    templates = ALL_TEMPLATES
    if template_id is not None:
        templates = [t for t in templates if t.template_id == template_id]
    return templates


def get_themes(template_id: str, theme: str | None = None) -> list[str]:
    """获取模板对应的 theme 列表。"""
    themes = THEMES.get(template_id, ["default"])
    if theme is not None:
        themes = [t for t in themes if t == theme]
    return themes


def expand_all() -> list[tuple[SequenceTemplate, str]]:
    """展开所有 (template, theme) 组合。"""
    result = []
    for t in get_templates():
        for theme in get_themes(t.template_id):
            result.append((t, theme))
    return result
