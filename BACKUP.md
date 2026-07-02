# Backup

This repo is local-only (no remote, by policy) or has unpushed work. It is backed up
as a git bundle to a private iCloud folder:

    ~/Library/Mobile Documents/com~apple~CloudDocs/code_workspace_backups/bundles/

- **Last bundle:** 2026-07-01 (`ssi_v2_2026-07-01.bundle`, all refs)
- **Refresh:** from the repo root run:
  ```sh
  git bundle create "$HOME/Library/Mobile Documents/com~apple~CloudDocs/code_workspace_backups/bundles/ssi_v2_$(date +%F).bundle" --all
  git bundle verify "$HOME/Library/Mobile Documents/com~apple~CloudDocs/code_workspace_backups/bundles/ssi_v2_$(date +%F).bundle"
  ```
  Delete bundles older than the two most recent.
- **Restore:** `git clone <bundle-file> <dir>`
- Bundles only contain *committed* work — commit before refreshing.
