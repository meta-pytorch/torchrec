# TorchRec Release Cut Instructions

## Step 1 — Switch to a Local Machine

These instructions should be run from a **local machine** (e.g., MacBook Pro), not from a devgpu or dev-server, since you'll need direct access to GitHub for pushing branches and triggering workflows.

## Step 2 — Clone / Navigate to the Repo

Navigate to the TorchRec repo at `~/local/torchrec`. If the directory doesn't exist, clone it first:

```bash
git clone https://github.com/meta-pytorch/torchrec/ ~/local/torchrec
cd ~/local/torchrec
```

## Step 3 — Fetch Latest and Check Branches

Fetch the latest info from GitHub and check existing branches:

```bash
git fetch --all
git branch -a
```

This ensures you see the most up-to-date remote branches before determining the next version.

## Step 4 — Confirm Release Details with User/Admin

Confirm with the user/admin:

- **Release cut date (range)** — when the cut should happen.
- **Version number** — typically the next increment from the highest existing release branch. For example, if the latest is `v1.4.0`, the next release would be `v1.5.0`.

## Step 5 — Find the Last Commit for the Cut

Find the last commit on or before the release cut date:

```bash
git log --oneline --before="<cut_date_plus_1_day>" -10
```

Identify the correct commit (no later than the cut date unless the user updates it).

Also provide the user a link to visually confirm commits in the browser:

```
https://github.com/meta-pytorch/torchrec/commits/main/?since=<cut_date_minus_1_week>&until=<cut_date_plus_1_week>
```

**Example for v1.5.0 (cut date 01/25/2026):**
- Command: `git log --oneline --before="2026-01-26" -10`
- Last commit: `b80a7ab9` (`#3689`)
- Browser link: https://github.com/meta-pytorch/torchrec/commits/main/?since=2026-01-17&until=2026-01-31

## Step 6 — Create the Release Branch

After confirming with the user, create a local release branch from the identified commit and switch to it:

```bash
git checkout -b release/<version> <commit_hash>
```

**Example:** `git checkout -b release/v1.5.0 b80a7ab9`

## Step 7 — Push the Branch to GitHub

Push the release branch to the remote:

```bash
git push origin release/<version>
```

**Example:** `git push origin release/v1.5.0`

## Step 8 — Bump Main Branch to Next Version

On a **devgpu/dev-server**, switch to the `main` branch and bump `version.txt` to the next pre-release version. This change should be submitted as a **diff** (not a GitHub PR).

```bash
git checkout main
git pull origin main
# Edit version.txt: e.g., 1.5.0a0 → 1.6.0a0
```

Create and submit a diff:

```bash
git add version.txt
jf submit -m "Bump version to <next_version>a0"
```

**Example for v1.5.0 release:** Update `version.txt` from `1.5.0a0` to `1.6.0a0`.

## Step 9 — Bump the Version on the Release Branch

Update `version.txt` from the pre-release version to the release version:

```
# Before (pre-release):
1.5.0a0

# After (release):
1.5.0
```

Commit the change with the message `update the release version to V<version>`, then confirm with the user to push the commit.

> **Note:** Claude cannot push directly to GitHub. The user must manually push the commit:
> ```bash
> git push origin release/<version>
> ```

## Step 10 — Run GitHub Workflows and Validate

The user needs to manually trigger the GitHub workflows at:

https://github.com/meta-pytorch/torchrec/actions

Specifically:
- **CPU unittests** — trigger and verify they pass. Use the **release channel** (or **test channel** as fallback).
- **CUDA unittests** — trigger and verify they pass. Use the **release channel** (or **test channel** as fallback).
- **Validate binaries** — ensure the release binaries are built and validated correctly. Use the **test channel**, because validate binaries checks if the `fbgemm-gpu` version matches `torchrec`, and only the test channel has the latest version since the release hasn't been promoted yet.

> **Important:** When triggering unittests, remind the user to select the **release channel** so that dependent libraries (FBGEMM, PyTorch, etc.) use their release versions rather than nightly/pre-release builds. If the release channel doesn't work (commonly due to FBGEMM not being available yet), the **test channel** can be used as a fallback. Validate binaries should always use the **test channel**.

### Verifying the Workflow Logs

Beyond checking that the workflows succeed, the user should also inspect the **raw logs** to verify the correct dependency versions were installed. Search the logs for `fbgemm-gpu` and `torch` install lines.

**What to look for:**

1. **Release channel URL** — The `--index-url` should be a release channel, e.g.:
   - `https://download.pytorch.org/whl/cpu` (CPU)
   - `https://download.pytorch.org/whl/cu129` (CUDA)
   - There should be **no** `test` or `nightly` suffix (e.g., `whl/test/cpu` or `whl/nightly/cpu` means the wrong channel was used).

2. **Dependency versions** — The installed versions of `torch` and `fbgemm-gpu` should match this release. Example log lines for v1.5.0:
   ```
   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cpu
   Successfully installed fbgemm-gpu-1.5.0+cpu numpy-2.3.5
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   Successfully installed torch-2.10.0+cpu
   ```

## Step 11 — Create a Release Candidate Tag

Create a release candidate tag following the naming pattern `v<version>-rc<N>` (e.g., `v1.5.0-rc1` for the first candidate):

```bash
git tag v<version>-rc1
```

If this is a subsequent release candidate (e.g., after fixes), increment the rc number (`-rc2`, `-rc3`, etc.).

Check existing tags to determine the next rc number:

```bash
git tag | grep v<version>
```

> **Note:** Claude cannot push directly to GitHub. The user must manually push the tag:
> ```bash
> git push origin v<version>-rc1
> ```

**Example:** `git tag v1.5.0-rc1 && git push origin v1.5.0-rc1`

## Step 12 — Draft Release Notes

Generate release notes and review them interactively with the user.

### 12.1 — Gather Commits

Gather all commits between the previous release and the current one:

```bash
git log --oneline origin/release/<previous_version>..release/<version>
```

Also gather author information to identify feature owners:

```bash
git log --oneline --format="%h %s | %an" origin/release/<previous_version>..release/<version>
```

### 12.2 — Draft Initial Release Notes

Organize the commits into the following structure:

```markdown
## Announcement
<!-- Optional: upcoming changes, deprecations, migration notices -->

## New Features
### <Feature Name>
<Short paragraph describing the feature, its motivation, and impact.>
* <Item description> [#PR1, #PR2]
* <Item description> [#PR3]

### <Another Feature>
...

## Change Log
* <Item> [#PR]
* <Item> [#PR1, #PR2]
* [full change log](https://github.com/meta-pytorch/torchrec/compare/release/<previous_version>...release/<version>)

### compatibility
* fbgemm-gpu==<version>
* torch==<version>

### test results
<!-- Add test result screenshot after Step 9 workflows pass -->
```

**Guidelines for initial draft:**
- Group related commits under descriptive subsections in **New Features**.
- Each subsection should have a short summary paragraph followed by bullet items with PR references (e.g., `[#1234]`).
- Consolidate related commits into single items where possible (e.g., multiple PRs for the same feature).
- Smaller improvements, bug fixes, and miscellaneous changes go in **Change Log**.
- Use `git log --format="%h %s | %an"` to identify the main author of each feature area.

### 12.3 — Interactive Review with User

Walk through each New Features section with the user and ask for feedback:

- **Categorization** — Should items be promoted to New Features or demoted to Change Log?
- **Consolidation** — Should related items be merged into a single line?
- **Missing commits** — Check by author (`git log --author="<name>"`) to find commits that may be missing.
- **Descriptions** — Are the descriptions accurate? The user or feature owners may provide better context.
- **Naming** — Are the section/item names appropriate?

### 12.4 — Finalize

- Confirm **compatibility** versions (`fbgemm-gpu` and `torch`) with the user.
- Add **test results** screenshot after Step 9 workflows pass.
- Save the release notes to `docs/release_notes_<version>.md`.

**Example:** For v1.5.0, commits were gathered with:
```bash
git log --oneline origin/release/v1.4.0..release/V1.5.0
```

## Step 13 — Draft Workplace Release Post

Draft a release announcement post for the TorchRec FYI Workplace group (see `docs/release_post_v1.x.0.md` for examples).

The post should follow this structure:

1. **Opening** — Announce the release with a link to the GitHub release page, and mention coordination with PyTorch and FBGEMM releases.
2. **Supported versions** — List supported Python versions and CUDA versions.
3. **Download links** — download.pytorch.org (nightly and stable) and PyPI (stable).
4. **Announcement** — Any upcoming changes or migration notices (optional).
5. **New Features** — Condensed summary of each feature section from the release notes (1–2 sentences each, numbered list).
6. **Other Notable Changes** — Brief bullet list of items from the Change Log.
7. **Link to full release notes** — Point to the GitHub release page.
8. **Acknowledgments** — Use the following defaults and ask the user to confirm or update:
   - `#thanks Supadchaya Puangpontip, Gantaphon Chalumporn, and Benson Ma for coordinating the release from FBGEMM side`
   - `#thanks Andrey Talman for promoting the release from pytorch dev-infra side`
   - Ask the user if there are additional contributors to acknowledge.
9. **Release schedule** — Mention the current release cut date and the next expected release cut (~3 months later, quarterly cadence).

Save the draft post to `docs/release_post_<version>.md` and review with the user before posting.

## Step 14 — Promote the TorchRec Build

Ask the PyTorch infra team to promote the TorchRec build from the test channel to the stable/release channel.

1. Clone or navigate to the [pytorch/test-infra](https://github.com/pytorch/test-infra) repository.
2. Update `TORCHREC_VERSION` in two files:
   - `release/promote.sh` — update `TORCHREC_VERSION` to the new version (e.g., `1.4.0` → `1.5.0`)
   - `release/release_versions.sh` — update `TORCHREC_VERSION` to the new version (e.g., `1.4.0` → `1.5.0`)
3. Create a PR with:
   - **Title:** `[test-infra] update TorchRec release version to v<version>`
   - **Summary:** Include a link to the release tag, a screenshot of passing CI workflows from Step 10, and verification of `torchrec.__version__` output.
4. Contact the PyTorch dev-infra team to review and merge:
   - **Primary PoC:** Andrey Talman
   - **Backup:** Huy Do

**Example PR:** https://github.com/pytorch/test-infra/pull/7557
