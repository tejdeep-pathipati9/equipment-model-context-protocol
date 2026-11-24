#!/usr/bin/env bash
# PostToolUse hook: run the unit test suite after Claude edits a Python file.
# Exit 2 on failure so the test output is fed back to Claude to fix.

file=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')

# Only react to Python source/test changes.
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# Nothing to run until the project is scaffolded (step 2 of PLAN.md).
[ -f pyproject.toml ] || exit 0

output=$(uv run pytest -m "not eval" -q 2>&1)
status=$?

# 0 = passed, 5 = no tests collected yet: both fine.
if [ "$status" -eq 0 ] || [ "$status" -eq 5 ]; then
  exit 0
fi

echo "pytest failed after editing $file:" >&2
echo "$output" | tail -n 60 >&2
exit 2
