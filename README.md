# duc-semantic
This repository hosts the contributor source files for the duc-semantic model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).
## meta
| | |
|-|-|
| id | 8154347c-a642-4e9e-91f1-ba800c0b532f | 
| application_area | Computer Vision | 
| task | Segmentation | 
| task_extended | Semantic Segmentation | 
| data_type | Image/Photo | 
| data_source | https://www.cityscapes-dataset.com/ | 
## publication
| | |
|-|-|
| title | Understanding Convolution for Semantic Segmentation | 
| source | arxiv | 
| url | https://arxiv.org/abs/1702.08502 | 
| year | 2018 | 
| authors | Panqu Wang, Pengfei Chen, Ye Yuan, Ding Liu, Zehua Huang, Xiaodi Hou, Garrison Cottrell | 
| abstract | Recent advances in deep learning, especially deep convolutional neural networks (CNNs), have led to significant improvement over previous semantic segmentation systems. Here we show how to improve pixel-wise semantic segmentation by manipulating convolution-related operations that are of both theoretical and practical value. First, we design dense upsampling convolution (DUC) to generate pixel-level prediction, which is able to capture and decode more detailed information that is generally missing in bilinear upsampling. Second, we propose a hybrid dilated convolution (HDC) framework in the encoding phase. This framework 1) effectively enlarges the receptive fields (RF) of the network to aggregate global information; 2) alleviates what we call the gridding issue caused by the standard dilated convolution operation. We evaluate our approaches thoroughly on the Cityscapes dataset, and achieve a state-of-art result of 80.1% mIOU in the test set at the time of submission. We also have achieved state-of-theart overall on the KITTI road estimation benchmark and the PASCAL VOC2012 segmentation task. Our source code can be found at https://github.com/TuSimple/TuSimple-DUC. | 
| google_scholar | https://scholar.google.com/scholar?oi=bibs&hl=en&cites=14464615562378306377 | 
| bibtex | @article{Wang2018UnderstandingCF, title={Understanding Convolution for Semantic Segmentation}, author={Panqu Wang and Pengfei Chen and Ye Yuan and Ding Liu and Zehua Huang and Xiaodi Hou and Garrison W. Cottrell}, journal={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)}, year={2018}, pages={1451-1460}} | 
## model
| | |
|-|-|
| description | DUC is a semantic segmentation model. It is benchmarked using the mIOU (mean Intersection Over Union) score and can be used in any application requiring semantic segmentation. It is trained on the cityscapes dataset which contains images from urban street scenes. Hence, it can be used in self driving vehicle applications. | 
| architecture | Convolutional Neural Network (CNN) | 
| learning_type | Supervised learning | 
| format | .onnx | 
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) | 
| license | model license can be viewed [here](contrib_src/license/model) | 
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).

