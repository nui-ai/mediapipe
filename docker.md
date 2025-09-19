## The Ubuntu 24.04 Docker Image

+ This repository's Dockerfile has two roles:
  1. and for reproducing the mediapipe build in a clean container ― this shows reproducibility in building mediapipe as no host system dependencies/configurations are involved.
  2. It was also adapted for use as a GitHub Actions self-hosted runner, but that part of it is commented now, as it failed on the github side without proper error messages.
 
+ The GitHub Actions runner binary is downloaded and extracted in the image as a non-root user, but registration and execution must be performed at runtime.

### Steps to Launch a Self-Hosted Runner

1. **Build the Docker image (if not already done):**
   ```bash
   docker build --no-cache -t mediapipe-build .
   ```

2. **Start a container interactively as the `runner` user:**
   ```bash
   docker run -it --name mediapipe-gh-runner \
     -v /var/run/docker.sock:/var/run/docker.sock \
     -v "$PWD":/mediapipe \
     --user runner \
     mediapipe-build
   ```
   - The `--user runner` flag ensures you are not running as root (required by GitHub runner setup).
   - The `-v /var/run/docker.sock:/var/run/docker.sock` is optional, only needed if you want workflows to run Docker commands inside the runner.

3. **Register the runner (inside the container):**
   - Get a registration token for your repo at [https://github.com/nui-ai/mediapipe/settings/actions/runners](https://github.com/nui-ai/mediapipe/settings/actions/runners).
   - Then run:
     ```bash
     cd /mediapipe/actions-runner
     ./config.sh --url https://github.com/nui-ai/mediapipe --token <YOUR_TOKEN>
     ```

4. **Start the runner (inside the container):**
   ```bash
   ./run.sh
   ```

- The runner will now listen for jobs from GitHub Actions and execute them inside the container with all necessary dependencies and environment setup.

**Note:**  
Do not run the `config.sh` or `run.sh` as root, or you will get the error `Must not run with sudo`.

For detailed, stepwise instructions see [SETUP_SELF_HOSTED_RUNNER.md](SETUP_SELF_HOSTED_RUNNER.md).