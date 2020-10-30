# lightfield-3dcv

## Requirements
Make sure to download all required packages first, i.e. run:
```
pip install -r requirements.txt
```
as well as download the dataset from [4D Light Field Dataset](https://lightfield-analysis.uni-konstanz.de/) and put it in the folder `data` in the root of this repository.

## Usage
All of our code is usable using `Jupyter Notebooks`. Starting a server and accessing `Train_Model.ipynb` one can train a model from scratch. With no changes made the model is trained on `128 x 128` pixel images, which are randomly cropped from the dataset consisting of `512 x 512` pixel images on a GPU or CPU. For better performance parameters like `batch_size`, `input_size`, `num_workers` and `hidden_dims` can be changed. A learning rate between `1e-3` and `1e-5` showed the most sucess with the later giving the edge in best reconstruction after more than 6000 epochs or as finetuning after a good pretraining.

To use the `View_Synthesis` one can download a pretrained model from [our Website](https://uni-heideiberg.de/final.pth) or use an own made model. One can choose the `path` to the model as well as the options to use `RandomCrop` or `Resize` to synthesize images. If one wants to use CPU only, 5 lines of code in the loader part have to be commented out.
