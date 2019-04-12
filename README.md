### Visual Motif Removal
Source code for the <a href="https://arxiv.org/abs/1904.02756" target="_blank">paper</a> Blind Visual Motif Removal from a Single Image.


<p align="center" style= "cursor: text;">
<a href="###" style= "cursor: text;"><img style= "cursor: text;" src="http://www.pxcm.org/motif/arch_diagram.png"></a>
</p>

**Prerequisites**
* <a href="https://pytorch.org/" target="_blank">Pytorch</a> ≥ 0.4
* Coco pythonApi: <a href="https://github.com/cocodataset/cocoapi" target="_blank">for python 2.7</a> or <a href="https://github.com/philferriere/cocoapi" target="_blank">for python 3</a> (optional for images datasets).


A pre-trained semi-transparent emojis removal model is available by running the script:  *demo / run_demo.py*.
 
#### Training

Start a training session, by run the file *train / train_main.py*.<br>
Different training configurations are placed at the top.

**Paths configurations**
* *root_path* – the main data path.
* *train_tag* – your network name. The checkpoint folder will be named after this tag.
* *cache_root* - list of directories with prepared training and test images. See _Create new Datasets_ section for more information. 
 
**Network configurations**
* _num_blocks_ – number of residual blocks between each layer.
* _shared_depth_ – shared layers between the decoders.
* _use_vm_decoder_ – If True, the network will contain a motif decoder branch.  

#### Testing
The _utils / visualize_utils.py_ script may assist in order to run a trained network on different images.
The _root_path_ and _train_tag_ from above should be defined on top.


#### Datasets
**Images** <br>
The _data_prep / coco_download.py_ script might be helpful to download a collection of images from <a href="http://cocodataset.org/#home" target="_blank">Microsoft COCO dataset</a>.<br>

**Text Motifs** <br>
The visual Motifs may be generated from a text file. examples of the text format are found at  _data / text_ folder or use 
the _split_text.py_ script on a row text file.

**Create new Dataset** <br>
To create a training data use the file _utils / cache_utils.py_. In there you will define the dataset configurations:
* _dataset_tag_- name for the dataset
* _images_root_ – path to a background images folder.
* _cache_root_- where should the data be saved.
* vm_root – path to the motifs dataset. May lead to: <br>
  - Motif image/s file or folder.
  - Text file (.txt), as described at the previous item.

  
  
