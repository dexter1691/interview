 ## Note:
- I am leaving out code to plot loss curves on tensorboard, because this didn't seem necessary for this exercise.
- While working on a notebook might not be suitable for big projects, I thought for this exercise, it gives a nice way to look through the training code in a concise manner.
- I just used a small subset of flickr30K dataset to show how to learn the Sobel kernel. 

## How to run.
The setup instructions are fairly straightforward. 

Create a conda environment `conda create -n poly` and activate `conda activate poly`.

Install libraries through conda. Note my machine has a CUDA 11.3 installed. Change the installation instructions to match your CUDA toolkit/drivers.
`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`

Install a few other libraries `pip install datasets opencv-python jupyterlab`

Most importantly, add this environment into ipython kernels so that you can use it in Jupyter lab.
```
pip install ipykernel
python -m ipykernel install --user --name=poly
```

To look at the code, simply run `jupyter lab`, and open `learn_kernel.ipynb` notebook with the `poly` ipython kernel.

## Directory structure
```
|- README.md
|- learn_kernel.ipynb # Starting point. Contains the training code.
|- utils.py # contains the utility function for visualizing, and creating data.
|- model.py # contains the model definition.
|- checkpoints/ # contains the checkpoints for the model.
|- data/ # contains the data.
|- .gitignore
```
