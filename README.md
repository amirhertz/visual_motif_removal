## Deep watermark separation
<!-- <p align="center" style= "cursor: text;">
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/watermarks/teaser2.gif"></a>
</p> -->
This is a school project which demonstrates the use of deep neural network to separate a target image from a disturbing watermark overlay.
This work is based on the paper <a href="https://arxiv.org/abs/1801.04102" target="_blank">"generative single image reflection separation technique"</a>, Donghoon Lee, Ming-Hsuan Yang, Songhwai Oh. Their article deals with a similar problem in which a single image contain a transmitted and a reflected scene, and with the aid of a neural network, this image can be separated into the two different semantic images.

The following sections guide through some technical aspects of the code; A detailed overview of the method is found <a href="http://www.pxcm.org/watermarks/watermark_removal.pdf" target="_blank">in here</a>.

**Prerequisites**
* <a href="https://pytorch.org/" target="_blank">Pytorch</a> ≥ 0.4
* Coco pythonApi: <a href="https://github.com/cocodataset/cocoapi" target="_blank">for python 2.7</a> or <a href="https://github.com/philferriere/cocoapi" target="_blank">for python 3</a> (optional for images datasets).

### Training
Start a training session, by run the file **train / train_main.py**.<br>
<!-- <p align="center" style= "cursor: text;">
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/watermarks/fig_b_baseline.png"></a>
</p> -->
Different training configurations are placed at the top of the file.

#### Flow configurations
May be set to *train*, *test* or *evaluate*
* *train* – start a training session.
* *test* – load a pre-trained network and save screenshot of the the network performance on a batch of the test dataset images.

[//]: # (*evaluate* – load a pre-trained network and print its PSNR and DSSIM scores on test / train datasets.)


#### Paths configurations
* *root_path* – the main path of the data folder.
* *image_domain* – unique name for the images dataset. 
* *wm_tag* – unique name for the watermarks dataset. 
* *train_tag* – your network name. The checkpoint folder will be named after this tag, together with the _image_domain_ and _wm_tag_.
* *cache_root* - list of directories with prepared training and test images. It will be used when _from_cache is True_.
   See _Create new Datasets_ for more information. 

 
#### Network configurations
The baseline network is based on the Unet architecture and contains one encoder and two decoders.
The first image-decoder is responsible for retrieving the ground truth pixels at the watermark area. The second mask-decoder is responsible for deciding in which pixels lays the watermark.<br>

Some parameters that may be configured:
* _num_blocks_ – number of convolution blocks between each encoder / decoder layer.
* _residual_ – boolean. If true, the blocks will be used in a residual manner.
* _transfer_data_ – boolean. If true, each encoded layer will be transferred to the 	suitable decoder layer.
* _concat_ – boolean. Define the way in which the up-conv data is combined with the down-conv data. Could be by concatenate or addition. 
* _patch_size_ – integer. Define the size of the patcehs to load.

<!-- <p align="center" style= "cursor: text;">
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/watermarks/fig_a_unet.png"></a>
</p>-->

#### Train configurations
* _gamma1_ – the coefficient for the l1 loss applied to the reconstructed pixels. 
* _gamma2_ – the coefficient for the l1 loss applied to the reconstructed watermark.
* _epochs_ – number of training epochs over the images dataset.
* _batch_size_ – number of training examples in each iteration.
* _print_frequency_ – number of iterations between the loss prints.
* _save_frequency_ – number of epochs between network checkpoints.
* _device_ – torch.device.

#### Upgrading options
* _shared_depth_ – shared layers between the decoders. could be up to 4.
* _use_wm_decoder_ – boolean. If True, the network will contain a watermark decoder branch, dedicated for generating the ground truth watermarks.  

<!-- <p align="center" style= "cursor: text;">
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/watermarks/fig_c_baseline_upgrades.png"></a>
</p>-->

### Datasets
At _data_prep_ folder, are supplied two utility scripts for an initial dataset.
#### Images
The _coco_download.py_ script might be helpful to download a collection of ground truth images from <a href="http://cocodataset.org/#home" target="_blank">Microsoft COCO dataset</a>.<br>
Downloading parameters are:
* _data_dir_ – path to the main directory of the images datasets.
* _set_name_ – name for the images collection. Will be used as a the collection folder name inside _data_dir_.
* _categories_ – list of images categories to download from (possible categories are listed at the top of the file). 
* _download_count_ – number of images to download 

#### Text watermarks
As mentioned before, the watermarks may be generated from a Text file (.text).
The text will be splitted into separated watermarks with whitespace as delimiter.
Within each single watermark, \<sp> and \<br> tags are used as whitespace and newline respectively.<br>
At _data / text_, are supplied two text files for single and multiple watermarks.
The _split_text.py_ script could assist to create a custom text watermarks file.
This script receives a text file and adds \<sp> and \<br> tags to split it randomly into a watermarks collection.
Its parameters are:
* _MAX_LENGTH_ – max length of characters within a single watermark
* _UNWANTED_SIGNS_ – set of unwanted signs within the watermarks, e.g. different punctuation marks.
* _input_file_ – path to the raw text file.
* _output_file_ – path to the new file which is created.

<!-- <p align="center" style= "cursor: text;">
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/watermarks/words_animals.gif"></a>
</p> -->

### Create new Datasets
In order to prepate cache datesets use the file _utils.cache_utils.py_. In there you will define the dataset parameter:

* _cache_root_- where should the data be saved. this directory will be given at the training session.
* _images_root_ – path to the images dataset.
* wm_root – path to the watermarks dataset. May lead to: <br>
  - A single watermark-image file.
  - Folder witch contains multiple watermarks.
  - Text file (.txt), as described at the previous item.
* _image_size_ – during train, the images will be cropped and resize to this size.
* _watermark_size_ – pair (min size, max size) of the watermark size range.
* _rotate_wm_ / _scale_wm_ / _crop_wm_ – booleans for train data augmentation.
* _batch_wm_ – integer. Max number of watermarks to spread in the train dataset. 
* _weight_ – pair (min weight, max weight) of the watermark opacity range.
* _use_rgb_ – boolean/string. If false, the watermark will be filled with white color. If 'gray' the watermark will be grayscele. 
* _perturbate_ – boolean. If true, a random perturbation smoothed field will be applied to the watermark.
* _opacity_var_ – double (0 to 1). Adds an opacity variance around the weight of the watermark. The opacity field is being smoothed before applying to the watermark.
* _font_ – directory of fonts to choose from (when text wm is used).
* _text_border_ – integer. max border line width (when text wm is used).
* _num_train_ / _num_test_ – integers. train / test datsets sizes. 

### Visualize
At _utils / visualize_utils.py_ there is a script which assist to run a trained network on different images.
At the top of the file should be defined: the _root_path, image_domain, wm_tag and train_tag_, same as were configured during training.
Afterwards, the _resources_root_ path should be set and contains custom test images.
The script may be run on _'loader'_ or _'net'_ mode (by setting the _get_loader_or_net_images_ variable).

#### Loader mode
A batch of test images will be saved in the resources folder.
For each synthesized image, an additional data will be saved: the ground truth image, ground truth watermark and the watermark mask.
The batch_size, image_size and watermark_size of the loader may be reconfigured.

#### Net mode
The trained network will be loaded and run on the images inside the resources folder (could be synthesized images from the loader mode or any other images to try on)
For each test image, the outputs would be reconstructed image, reconstructed mask and reconstructed watermark (if the network contains a watermark branch).

<!-- <p align="center" style= "cursor: text;">
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/watermarks/sports_emojis.gif"></a>
</p> -->
