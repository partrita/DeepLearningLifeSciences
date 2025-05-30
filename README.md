# Deep Learning for Life Sciences

Example code from the book **"Deep Learning for the Life Sciences"**. This repository provides hands-on examples applying deep learning to problems such as drug discovery, genomics, and protein structure prediction[1].

---

## Features

- Example code and tutorials from the book
- Ready-to-use Docker environment with DeepChem 2.4.0
- GPU acceleration support (NVIDIA)

---

## Installation

### 1. Docker Image (Recommended)

You can use the pre-built Docker image from DockerHub:

- **Image:** `deepchemio/deepchem:2.4.0`
    - Built using conda (DeepChem version 2.4.0)
    - Dockerfile is located in the `docker/tag` directory

#### Install Docker

Follow the official [Docker installation guide](https://docs.docker.com/engine/install/).

---

### 2. Clone This Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/partrita/DeepLearningLifeSciences.git
cd DeepLearningLifeSciences
```

---

## Usage

### 1. Build the Docker Image Locally

If you want to build your own image (for example, after modifying code):

```bash
docker build -t deeplife .
```

### 2. Run the Docker Container

Mount your current directory into the container for easy file access:

```bash
docker run --rm -it -v $PWD:/root/mydir/ deeplife
```

#### GPU Support

- **If `nvidia-docker` is installed:**
    ```bash
    nvidia-docker run --rm -it -v $PWD:/root/mydir/ deeplife
    ```
    or
    ```bash
    docker run --runtime nvidia --rm -it -v $PWD:/root/mydir/ deeplife
    ```
- **If `nvidia-container-toolkit` is installed:**
    ```bash
    docker run --gpus all --rm -it -v $PWD:/root/mydir/ deeplife
    ```

### 3. Using the Official DeepChem Image

You can also use the official DeepChem image directly:

```bash
docker run --rm -it -v $PWD:/root/mydir/ deepchemio/deepchem:2.4.0
```

---

## References

- [Deep Learning for the Life Sciences (O'Reilly)](https://www.oreilly.com/library/view/deep-learning-for/9781492039822/)
- [DeepChem Documentation](https://deepchem.io/)
- [Docker Documentation](https://docs.docker.com/)

## License

MIT License. See [LICENSE](LICENSE) for details.
