# DeepLearningLifeSciences

Example code from the book "Deep Learning for the Life Sciences"

# Installation

I recommend to using a docker, you can pull images from DockerHub.

- deepchemio/deepchem:2.4.0
  - Image built by using a conda(2.4.0 is a version of deepchem)
  - This image is built when we push 2.4.0. tag
  - Dockerfile is put in `docker/tag`_ directory

## How to install docker?

Check this officical docker [website](https://docs.docker.com/engine/install/).

## Clone the git repo

First, you need to clone this repo in your computer.

```bash
gh repo clone partrita/DeepLearningLifeSciences
```

Then, you create a container based on the image.

```bash
docker run --rm -it -v $PWD:/root/mydir/ deepchemio/deepchem:2.4.0
```

### If you want GPU support:

#### If nvidia-docker is installed

```bash
nvidia-docker run --rm -it deepchemio/deepchem:latest
docker run --runtime nvidia --rm -it deepchemio/deepchem:latest
```

#### If nvidia-container-toolkit is installed

```bash
docker run --gpus all --rm -it deepchemio/deepchem:latest
You are now in a docker container which deepchem was installed. You can start playing with it in the command line.
```