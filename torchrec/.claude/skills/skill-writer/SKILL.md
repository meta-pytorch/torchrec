---
name: skill-writer
description: Guide users through creating Agent Skills for Claude Code. Use when the user wants to create, write, author, or design a new Skill for TorchRec, or needs help with SKILL.md files.
---

# TorchRec Skill Writer

This Skill helps you create well-structured Agent Skills for Claude Code specifically for the TorchRec project.

## When to use this Skill

Use this Skill when:
- Creating a new Agent Skill for TorchRec
- Writing or updating SKILL.md files
- Designing skill structure and frontmatter
- Converting existing TorchRec workflows into Skills

## Instructions

### Step 1: Determine Skill scope

First, understand what the Skill should do:

1. **Ask clarifying questions**:
   - What specific TorchRec capability should this Skill provide?
   - When should Claude use this Skill?
   - What tools or resources does it need?

2. **Keep it focused**: One Skill = one capability
   - Good: "sharding-optimizer", "embedding-config-validator"
   - Too broad: "distributed-training", "model-tools"

### Step 2: Choose Skill location

TorchRec Skills should be placed in:
```
fbcode/torchrec/.claude/skills/<skill-name>/SKILL.md
```

### Step 3: Create Skill structure

Create the directory and files:

```bash
mkdir -p fbcode/torchrec/.claude/skills/skill-name
```

For multi-file Skills:
```
skill-name/
├── SKILL.md (required)
├── reference.md (optional)
├── examples.md (optional)
└── templates/ (optional)
```

### Step 4: Write SKILL.md frontmatter

Create YAML frontmatter with required fields:

```yaml
---
name: skill-name
description: Brief description of what this does and when to use it
---
```

**Field requirements**:

- **name**:
  - Lowercase letters, numbers, hyphens only
  - Max 64 characters
  - Must match directory name
  - Good: `sharding-optimizer`, `kjt-validator`
  - Bad: `Sharding_Optimizer`, `KJT Validator!`

- **description**:
  - Max 1024 characters
  - Include BOTH what it does AND when to use it
  - Use specific trigger words users would say
  - Mention TorchRec concepts (embeddings, sharding, KJT, etc.)

**Optional frontmatter fields**:

- **allowed-tools**: Restrict tool access (comma-separated list)
  ```yaml
  allowed-tools: Read, Grep, Glob
  ```

- **argument-hint**: Hint for expected arguments
  ```yaml
  argument-hint: [feature or task description]
  ```

### Step 5: Write effective descriptions

The description is critical for Claude to discover your Skill.

**Formula**: `[What it does] + [When to use it] + [TorchRec keywords]`

**Examples**:

✅ **Good**:
```yaml
description: Optimize sharding plans for TorchRec embedding tables. Use when configuring DistributedModelParallel, analyzing sharding strategies, or tuning embedding performance.
```

✅ **Good**:
```yaml
description: Validate KeyedJaggedTensor (KJT) configurations and debug sparse tensor issues. Use when working with KJT, debugging embedding lookups, or validating feature configurations.
```

❌ **Too vague**:
```yaml
description: Helps with TorchRec
description: For distributed training
```

### Step 6: Structure the Skill content

Use clear Markdown sections:

```markdown
# Skill Name

Brief overview of what this Skill does for TorchRec.

## Quick start

Provide a simple example to get started immediately.

## Instructions

Step-by-step guidance for Claude:
1. First step with clear action
2. Second step with expected outcome
3. Handle edge cases

## TorchRec-Specific Patterns

Document TorchRec-specific patterns and conventions.

## Examples

Show concrete usage examples with TorchRec code.

## Best practices

- Key conventions to follow
- Common pitfalls to avoid
- When to use vs. not use

## Files to Reference

List important TorchRec files for context:
- `torchrec/distributed/` - Distributed training code
- `torchrec/modules/` - Core modules
```

### Step 7: Validate the Skill

Check these requirements:

✅ **File structure**:
- [ ] SKILL.md exists in correct location
- [ ] Directory name matches frontmatter `name`

✅ **YAML frontmatter**:
- [ ] Opening `---` on line 1
- [ ] Closing `---` before content
- [ ] Valid YAML (no tabs, correct indentation)
- [ ] `name` follows naming rules
- [ ] `description` is specific and < 1024 chars

✅ **Content quality**:
- [ ] Clear instructions for Claude
- [ ] TorchRec-specific examples provided
- [ ] Edge cases handled
- [ ] References to relevant TorchRec code

## TorchRec Skill Ideas

Here are some useful Skills to consider creating:

| Skill Name | Purpose |
|------------|---------|
| `sharding-optimizer` | Analyze and optimize embedding sharding plans |
| `kjt-validator` | Validate KeyedJaggedTensor configurations |
| `distributed-debug` | Debug distributed training issues |
| `embedding-benchmark` | Benchmark embedding performance |
| `migration-helper` | Help migrate to newer TorchRec APIs |

## Example: Complete TorchRec Skill

```yaml
---
name: sharding-analyzer
description: Analyze TorchRec sharding plans and suggest optimizations. Use when reviewing ShardingPlan, DistributedModelParallel configuration, or optimizing embedding distribution across devices.
---

# Sharding Analyzer

Analyze TorchRec sharding plans and suggest optimizations for embedding tables.

## Quick start

Run `/sharding-analyzer` on a file containing a ShardingPlan to get optimization suggestions.

## Instructions

1. Read the sharding plan configuration
2. Analyze table sizes and sharding strategies
3. Check for common anti-patterns:
   - Large tables with TABLE_WISE sharding
   - Small tables with ROW_WISE sharding
   - Unbalanced memory distribution
4. Suggest optimizations

## TorchRec Sharding Strategies

| Strategy | Best For | Avoid When |
|----------|----------|------------|
| TABLE_WISE | Small tables, < 1M rows | Large tables |
| ROW_WISE | Large tables, uniform access | Small tables |
| COLUMN_WISE | Wide embeddings, > 256 dim | Narrow embeddings |

## Files to Reference

- `torchrec/distributed/planner/` - Sharding planner
- `torchrec/distributed/sharding/` - Sharding implementations
```

## Output format

When creating a Skill, I will:

1. Ask clarifying questions about scope and requirements
2. Suggest a Skill name and location
3. Create the SKILL.md file with proper frontmatter
4. Include TorchRec-specific instructions and examples
5. Add references to relevant TorchRec code
6. Provide validation checklist
