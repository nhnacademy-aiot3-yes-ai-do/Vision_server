#!/usr/bin/env bash
set -euo pipefail

COVERAGE_RESULT="${COVERAGE_RESULT:-not_configured}"
COVERAGE_PERCENTAGE="${COVERAGE_PERCENTAGE:-not_available}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-not_available}"
SONAR_URL="${SONAR_URL:-not_available}"
COMMIT_MESSAGE="$(git log -1 --pretty=%B)"
IMAGE_NAME="${IMAGE_NAME:-not_available}"

case "$SIMULATE_FAILURE" in
  true|false) ;;
  *) SIMULATE_FAILURE=false ;;
esac

DISPATCH_FILE="$(mktemp)"
trap 'rm -f "$DISPATCH_FILE"' EXIT

jq -n \
  --arg image_name "$IMAGE_NAME" \
  --arg service_name "$SERVICE_NAME" \
  --arg repository "$SOURCE_REPOSITORY" \
  --arg branch "$SOURCE_BRANCH" \
  --arg commit_sha "$SOURCE_COMMIT_SHA" \
  --arg commit_message "$COMMIT_MESSAGE" \
  --arg actor "$SOURCE_ACTOR" \
  --arg quality_result "$QUALITY_RESULT" \
  --arg coverage_result "$COVERAGE_RESULT" \
  --arg coverage_percentage "$COVERAGE_PERCENTAGE" \
  --arg coverage_threshold "$COVERAGE_THRESHOLD" \
  --arg sonar_url "$SONAR_URL" \
  --arg build_result "$BUILD_RESULT" \
  --arg workflow_url "$SOURCE_WORKFLOW_URL" \
  --argjson simulate_failure "$SIMULATE_FAILURE" \
  -f ".github/scripts/dispatch-payload.jq" > "$DISPATCH_FILE"

gh api \
  --method POST \
  "repos/$CONFIG_REPOSITORY/dispatches" \
  --input "$DISPATCH_FILE"

echo "Config 중앙 배포 요청 완료: $CONFIG_REPOSITORY"
