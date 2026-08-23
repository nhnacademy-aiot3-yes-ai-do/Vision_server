{
  event_type: "deploy-service",
  client_payload: {
    service: {
      name: $service_name,
      repository: $repository,
      branch: $branch
    },
    commit: {
      sha: $commit_sha,
      message: $commit_message,
      actor: $actor
    },
    quality: {
      result: $quality_result,
      coverage_result: $coverage_result,
      coverage_percentage: $coverage_percentage,
      coverage_threshold: $coverage_threshold,
      sonar_url: $sonar_url
    },
    build: {
      result: $build_result,
      image_name: $image_name
    },
    source: {
      workflow_url: $workflow_url
    },
    options: {
      simulate_deploy_failure: $simulate_failure
    }
  }
}
