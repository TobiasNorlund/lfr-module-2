# Learning Feature Representations - Module 2

- Build and start a docker container:
```bash
./scripts/start.sh [--gpu] [--notebook] [--tensorboard] [-v|--mount /host/path:/container/path] [--detach]
```
- Start a development container in VS Code:
   There are two ways this can be done. 
   - Attach to the already running container (preferred when container is running on remote host)
      - In VS Code, install the `Remote-Containers` extention
      - Run `Remote-Containers: Attach to Running Container...` (F1). Select the newly created container
      - In the Explorer pane, click `Open Folder` and type the workspace directory (by default mounted to `/workspace`)
   - Let VS Code manage the container (preferred for local development)
      - In VS Code, install the `Remote-Containers` extention
      - Update `name` in `.devcontainer/devcontainer.json` to the value of `DOCKER_IMAGE_NAME`
      - Run `Remote-Containers: Reopen in container` (F1). Select the newly created container
