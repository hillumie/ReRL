pip install -U pip
pip install -r requirements_envpool.txt --no-cache-dir
pip install "jax[cuda12_pip]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-cache-dir
# pip install nvidia-cudnn-cu12==8.9.2.26