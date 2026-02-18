# FastFuels Git Workflow & Deployment Guide

## Overview

**NOTE: This is a set of guidelines, not law. Common sense and flexibility trump strict adherence to these rules.**

This guide outlines our team's Git workflow and deployment process. Our approach uses GitHub issues, feature branches, and automatic deployments to maintain development and production environments.

![Git Workflow Visualization](content/git_workflow.svg)

## Branch Structure

| Branch | Purpose | Environment | Notes                                |
|--------|---------|-------------|--------------------------------------|
| `main` | Production code | Production | Reflects code on `prod` environment. |
| `dev` | Integration & testing | Development | Reflects code on `dev` environment.  |
| `issue-XXX` | Feature/bug development | Local | One branch per GitHub issue.         |

## Workflow Steps

### Starting Work on a New Feature

1. **Create a GitHub issue** describing the feature or bug
2. **Create a new branch** from `main` (preferred) or `dev` if dependent on in-progress work
   ```bash
   # From main (preferred for independent features)
   git checkout main
   git pull
   git checkout -b issue-123-feature-name

   # OR from dev (for features dependent on in-progress work)
   git checkout dev
   git pull
   git checkout -b issue-123-feature-name
   ```
3. **Develop and commit** your changes
   ```bash
   git add .
   git commit -m "Implement feature X for #123"
   git push -u origin issue-123-feature-name
   ```

### Testing in Development Environment

1. **Create a Pull Request** to merge your issue branch into `dev`
2. **Merge to `dev`** when ready
3. **Automatic deployment** occurs to the development environment
4. **Test thoroughly** in the development environment

Note: Code reviews are not required for merging to `dev`, but are encouraged for complex changes.

### Deploying to Production

**Option 1: Deploy a Single Feature**
```bash
# After testing in dev, create PR from issue branch to main
# Create PR on GitHub: issue-123-feature-name → main
# Request review from at least one team member
```

**Option 2: Deploy Multiple Features**
```bash
# Deploy all tested features in dev to production
# Create PR on GitHub: dev → main
# Request review from at least one team member
```

After review approval and merging to `main`, automatic deployment to production occurs via GitHub Actions.

### Handling Hotfixes

1. **Create an issue branch** from `main` (using the same naming convention as regular issues)
   ```bash
   git checkout main
   git pull
   git checkout -b issue-456-critical-bug-fix
   ```
2. **Implement and test** the fix
3. **Create PRs** to merge into `main` AND `dev`
   - PR to `main` requires code review
   - PR to `dev` can be merged directly
4. **Merge to both branches** to ensure fix is in all environments

## Automatic Deployments

Each service has a GitHub Action workflow that:
1. Triggers when code is pushed to `main` or `dev`
2. Builds a Docker container image
3. Deploys to the appropriate environment:
   - `dev` branch → Development environment (`{service-name}-dev`)
   - `main` branch → Production environment (`{service-name}-prod`)

Server names follow the convention of `{service-name}-{environment}` (e.g., `treevox-dev` for development, `treevox-prod` for production).

## Best Practices

1. **Pull regularly** to stay up-to-date with team changes
2. **Create focused branches** that address a single issue
3. **Write descriptive commit messages** and reference issue numbers
4. **Keep PRs small and focused** for easier review
5. **Use meaningful branch names** with issue number and description
6. **Delete branches** after merging to keep the repository clean
7. **Don't force push** to shared branches (`main`, `dev`)
8. **Rebase long-lived branches** on latest `main` before final testing

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Deployment failed | Check GitHub Actions logs for error details |
| Merge conflict | Pull latest changes, resolve conflicts locally, then push |
| Need to undo a deployment | Use the previous container version from Artifact Registry |
| CI/CD pipeline error | Check GCP credentials and permissions |

## Quick Reference

```bash
# Start a new feature
git checkout main
git pull
git checkout -b issue-XXX-feature-name

# Add and commit changes
git add my_file_or_folder_with_changes
git commit -m "Implement feature X for #XXX"

# Push changes to remote
git push origin issue-XXX-feature-name

# Merge feature to dev

# Merge feature to main

# After PR is merged, clean up
git checkout main
git pull
git branch -d issue-XXX-feature-name
```
