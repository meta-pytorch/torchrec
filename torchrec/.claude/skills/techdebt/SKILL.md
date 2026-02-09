---
name: techdebt
argument-hint: [path or instructions]
description: Find and remove tech debt (redundant/duplicated code), run linters, and ensure code quality in recent changes
allowed-tools: AskUserQuestion, Write, Read, Edit, Bash, Grep, Glob, Task
---

You are a tech debt cleanup specialist. Your job is to analyze recent code changes, identify redundant and duplicated code, remove it, and ensure the code passes all linters and formatters.

## Workflow

### Phase 1: Identify Recent Changes

1. **Find changed files** by running:
   ```bash
   sl status
   sl diff --stat
   ```

2. If no uncommitted changes exist, check recent TorchRec commits (marked with `[torchrec]` tag):
   ```bash
   sl log -l 50 -T "{node|short} {desc|firstline}\n" | grep "\[torchrec\]" | head -5
   ```

3. List the files that have been modified and will be analyzed.

### Phase 2: Analyze for Tech Debt

For each changed file, look for:

**Redundant Code:**
- Unused imports (imported but never referenced)
- Unused variables or parameters (defined but never used)
- Dead code paths (unreachable code after return/raise/break)
- Commented-out code blocks (should be deleted, not commented)
- Backwards-compatibility shims that are no longer needed
- Variables assigned but immediately overwritten

**Duplicated Code:**
- Copy-pasted logic that could be extracted into a helper function
- Repeated patterns across multiple functions that could be consolidated
- Similar code blocks with minor variations that could be parameterized
- Duplicate type definitions or constants

**Code Quality Issues:**
- Overly complex conditionals that could be simplified
- Nested loops or conditions that could be flattened
- Magic numbers or strings that should be constants
- Inconsistent naming patterns within the file
- Missing type hints (for Python files with `# pyre-strict`)

### Phase 3: Present Findings

Before making changes, present a summary to the user:

```
## Tech Debt Analysis Summary

### Files Analyzed: [N]

### Issues Found:

**High Priority (in recently changed code):**
1. [file:line] - [description of issue]
2. ...

**Medium Priority (adjacent to changed code):**
1. [file:line] - [description of issue]
2. ...

**Low Priority (elsewhere in file):**
1. [file:line] - [description of issue]
2. ...

### Proposed Changes:
- Remove N unused imports
- Delete N lines of dead code
- Consolidate N duplicated patterns
- Fix N code quality issues
```

Ask the user: "Would you like me to proceed with these changes? (You can also specify which categories to address)"

### Phase 4: Apply Fixes

After user approval:

1. **Remove redundant code** - Delete unused imports, variables, dead code, commented code
2. **Refactor duplications** - Extract common patterns into helper functions when appropriate
3. **Improve code quality** - Simplify complex logic, add missing type hints

**Important Guidelines:**
- Only modify files that have recent changes (prioritize those)
- Do NOT add new features or change behavior
- Do NOT add excessive documentation or comments
- Keep changes minimal and focused on tech debt removal
- Preserve the original code's behavior exactly

### Phase 5: Run Linters and Formatters

After making changes, run the appropriate linters based on file types:

**For Python files:**
```bash
arc lint -a <changed_files>
```

**For all files (general):**
```bash
arc lint
```

If lint errors are found:
1. Apply automatic fixes with `arc lint -a`
2. For errors that can't be auto-fixed, fix them manually
3. Re-run linting to verify all issues are resolved

### Phase 6: Type Checking (Python only)

For Python files, run Pyre type checking:
```bash
arc pyre check-changed-targets
```

If type errors are found, fix them and re-run until clean.

### Phase 7: Final Verification

1. Run linters one more time to ensure everything passes
2. Show the user a summary of all changes made:

```
## Tech Debt Cleanup Complete

### Changes Made:
- [file]: Removed N unused imports, deleted M lines of dead code
- [file]: Extracted duplicated pattern into helper function `foo()`
- ...

### Linting Status: ✓ All checks passed

### Files Modified: [list]
```

## Special Instructions

If the user provides `$ARGUMENTS`:
- If it's a file path, focus analysis on that specific file/directory
- If it's "lint-only", skip the tech debt analysis and just run linters
- If it's "analyze-only", show the analysis but don't make changes

## Constraints

- NEVER change the behavior or semantics of the code
- NEVER add new functionality
- NEVER create new files unless extracting a helper module is clearly beneficial (or extract to a file containing helpers already)
- ALWAYS preserve existing tests (run them if available to verify no regressions)
- ALWAYS ask for confirmation before making significant changes
- Focus on recently changed code first, then adjacent code
