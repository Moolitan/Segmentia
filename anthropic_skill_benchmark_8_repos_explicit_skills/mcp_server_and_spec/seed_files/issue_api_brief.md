# Issue API Brief

Goal:
Build an MCP server for a simple issue-tracking service.

Available core REST APIs:
- GET /issues/{id}                  fetch a single issue
- GET /issues?status=&assignee=     filter issues
- POST /issues                      create an issue
- POST /issues/{id}/comments        add a comment to an issue

Minimum requirements:
1. The agent can query issues
2. The agent can create issues
3. The agent can add comments to issues

Constraints:
- No attachment upload for now
- No bulk modification for now
- No user-permission integration for now
