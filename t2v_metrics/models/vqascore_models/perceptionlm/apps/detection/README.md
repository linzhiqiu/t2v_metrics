# Object Detection with PE

## Getting started

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation instructions.

## Results and Fine-tuned Models


### LVIS

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">detector</th>
<th valign="bottom">vision encoder</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_PEcore_G_lvis75ep -->
 <tr><td align="left"><a href="projects/ViTDet/configs/LVIS/mask_rcnn_PEcore_G_lvis75ep.py">Mask R-CNN</a></td>
<td align="center">PE core G</td>
<td align="center">51.9</td>
<td align="center">47.9</td>
<td align="center"><a href="https://huggingface.co/facebook/PE-Detection/resolve/main/mask_rcnn_PEcore_G_lvis75ep.pth">model</a></td>
</tr>
<!-- ROW: mask_rcnn_PEspatial_G_lvis75ep -->
 <tr><td align="left"><a href="projects/ViTDet/configs/LVIS/mask_rcnn_PEspatial_G_lvis75ep.py">Mask R-CNN</a></td>
<td align="center">PE spatial G</td>
<td align="center">54.2</td>
<td align="center">49.3</td>
<td align="center"><a href="https://huggingface.co/facebook/PE-Detection/resolve/main/mask_rcnn_PEspatial_G_lvis75ep.pth">model</a></td>
</tr>
</tbody></table>


### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">detector</th>
<th valign="bottom">vision encoder</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_PEcore_G_coco75ep -->
 <tr><td align="left"><a href="projects/ViTDet/configs/COCO/mask_rcnn_PEcore_G_coco75ep.py">Mask R-CNN</a></td>
<td align="center">PE core G</td>
<td align="center">57.0</td>
<td align="center">49.8</td>
<td align="center"><a href="https://huggingface.co/facebook/PE-Detection/resolve/main/mask_rcnn_PEcore_G_coco75ep.pth">model</a></td>
</tr>
<!-- ROW: mask_rcnn_PEspatial_G_coco36ep -->
 <tr><td align="left"><a href="projects/ViTDet/configs/COCO/mask_rcnn_PEspatial_G_coco36ep.py">Mask R-CNN</a></td>
<td align="center">PE spatial G</td>
<td align="center">57.8</td>
<td align="center">50.3</td>
<td align="center"><a href="https://huggingface.co/facebook/PE-Detection/resolve/main/mask_rcnn_PEspatial_G_coco36ep.pth">model</a></td>
</tr>
</tbody></table>


### Training
By default, we use 64 GPUs in slurm training, for example

```
sbatch scripts/coco/train_mask_rcnn_PEspatial_G_coco36ep.sh
```

### Evaluation
Evaluation is running locally
```
bash scripts/evaluate_local.sh --config-file projects/ViTDet/configs/COCO/mask_rcnn_PEspatial_G_coco36ep.py train.output_dir="/path/to/output_dir" train.init_checkpoint="/path/to/mask_rcnn_PEspatial_G_coco36ep.pth"
```


## SOTA COCO Object Detection

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">detector</th>
<th valign="bottom">vision encoder</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">box(TTA)<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: DETA -->
 <tr><td align="left">DETA</td>
<td align="center">PE spatial G</td>
<td align="center"> 65.2 </td>
<td align="center"> 66.0 </td>
<td align="center"><a href="https://huggingface.co/facebook/PE-Detection/resolve/main/deta_coco_1824pix.pth">model</a></td>

</tr>
</tbody></table>

More details are in [DETA_pe](DETA_pe)


## Acknowledgment

This code is built using [detectron2](https://github.com/facebookresearch/detectron2) and [DETA](https://github.com/jozhang97/DETA).
