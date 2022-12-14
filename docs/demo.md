# Demo
Run Sparse Plane Reconstruction on a pair of input images.
Download our pretrained model [model_ICCV.pth][1] and save to `sparsePlane/models`
```bash
cd sparsePlane/models
wget https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/models/model_ICCV.pth
```
Inference on teaser images.
```
cd sparsePlane
python tools/inference_sparse_plane.py \
--config-file ./tools/demo/config.yaml \
--input ./tools/demo/teaser \
--output ./debug
```
- Predicted correpondence saved as `corr.png`:
![teaser corr](static/demo_corr.jpg)
- Reconstruction saved as `obj` file:
![obj](static/obj.png)

[1]: https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/models/model_ICCV.pth

Or you can inference on a list of image pairs by making a txt file
```
cd sparsePlane
python tools/inference_sparse_plane.py \
--config-file ./tools/demo/config.yaml \
--input ./tools/demo/images \
--img-list ./tools/demo/images/image_list.txt \
--output ./debug
```