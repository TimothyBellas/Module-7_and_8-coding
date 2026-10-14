# Push and verify the CI assignment

## 1. Put the files at the repository root

Extract `rag-docker.zip`. Inside its `rag-docker` folder, `.github/`,
`backend/`, and `frontend/` must be beside `docker-compose.yml`.
GitHub reads `.github/workflows/ci.yml` from the repository root.

If you already have this RAG project in a cloned Git repository, copy these
updated files into that project, preserving their relative paths:

- `.github/workflows/ci.yml`
- `backend/tests/test_api.py`
- `pytest.ini`
- `tests/test_configuration.py`
- `README.md`
- `CI-SETUP.md`
- `VALIDATION.md`

Open that repository folder in VS Code and use its PowerShell terminal.

## 2. Push to an existing repository

Check the branch and remote first:

```powershell
git status
git branch --show-current
git remote -v
```

On `main`, with your repository configured as `origin`, run:

```powershell
git add .github/workflows/ci.yml backend/tests/test_api.py pytest.ini tests/test_configuration.py README.md CI-SETUP.md VALIDATION.md
git commit -m "Add RAG CI workflow and endpoint tests"
git push origin main
```

If you are working on a feature branch, commit the same files, push that
branch with `git push -u origin HEAD`, and open a pull request into `main`.
The pull request triggers the same workflow.

## 3. If the project has never been pushed

Create an empty GitHub repository. Leave its README, license, and gitignore
options unchecked because the project already contains its own files.
From the extracted `rag-docker` folder, run the following. Replace the example
URL with the exact HTTPS URL shown on your new repository's page:

```powershell
git init -b main
git add .
git commit -m "Add containerized RAG application with CI"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Git may ask you to sign in to GitHub. `.env`, local virtual environments,
database data, and caches are excluded by `.gitignore`.

## 4. Verify in GitHub

1. Open your repository and select **Actions**.
2. Select **RAG CI**, then open the run for the commit you just pushed.
3. Wait for both **test** and **docker** to show green check marks.
4. Open **test** and inspect **Run backend endpoint tests and the existing suite**.
   It should report the three new tests plus the existing suite as passed.
5. Open **docker** and inspect **Verify both Docker images were created**.
   It should print image IDs and the `rag-backend:ci` and `rag-frontend:ci` tags.
6. Save the successful run's URL or take a screenshot showing both green jobs
   as assignment evidence.

If a job fails, open its red step and read the error before retrying. If there
is no run, confirm that the workflow is at the repository root, that you pushed
to `main` or opened a pull request, and that Actions is enabled for the
repository. A first pull request from a fork may require the repository owner's
approval before GitHub runs it.

## Optional local checks

Docker Desktop must be running to perform the same image checks on Windows:

```powershell
docker build --tag rag-backend:ci ./backend
docker build --tag rag-frontend:ci ./frontend
docker image inspect rag-backend:ci rag-frontend:ci
```

For local Python tests, use Python 3.11, matching the workflow and Dockerfiles:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt -r frontend/requirements.txt -r tests/requirements.txt
.\.venv\Scripts\python.exe -m pytest backend/tests tests
```

Local tests are optional for launching GitHub Actions: GitHub installs Python
and dependencies itself. Passing local tests alone does not establish that the
Docker builds or the GitHub workflow succeeded.
