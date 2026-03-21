#!/usr/bin/env bash
# scripts/setup_github_runner.sh — Setup a self-hosted runner on macOS ARM.
# This is required for testing large models that OOM on GitHub-hosted runners.

set -euo pipefail

RUNNER_VERSION="2.316.1" # latest as of May 2024
PLATFORM="osx-arm64"

echo "Setting up GitHub Actions runner ($PLATFORM)..."

# 1. Create directory
mkdir -p actions-runner && cd actions-runner

# 2. Download
if [[ ! -f "actions-runner-osx-arm64-$RUNNER_VERSION.tar.gz" ]]; then
    curl -o "actions-runner-osx-arm64-$RUNNER_VERSION.tar.gz" -L \
        "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-osx-arm64-$RUNNER_VERSION.tar.gz"
fi

# 3. Extract
tar xzf "./actions-runner-osx-arm64-$RUNNER_VERSION.tar.gz"

echo "✅ Runner downloaded and extracted."
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/matt-k-wong/mlx-flash/settings/actions/runners/new"
echo "2. Copy the config command (./config.sh --url ... --token ...)"
echo "3. Run it in the 'actions-runner' directory."
echo "4. Use labels: [self-hosted, macos, arm64]"
echo "5. Start the runner: ./run.sh"
