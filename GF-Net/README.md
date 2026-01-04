# GatedFusion-Net
We introduce GatedFusion-Net (GF-Net), built on the SegFormer Transformer backbone, as the first architecture to unify RGB, depth (D), infrared intensity (I), thermal (T), and ultraviolet (UV) imagery for dense semantic segmentation on the MM5 dataset.

## Train
1. Pretrain weights:

    Download the pretrained weights here [pretrained GF-Net](https://drive.google.com/drive/folders/1Ea-OFVtBz5uzMSFG7sYCOjvq9i5uhYqc?usp=sharing).

2. Dataset

    Download dataset from here: [MM5 on figshare](https://doi.org/10.6084/m9.figshare.28722164)
    
3. Config

    Edit config file in `configs/`, including dataset and network settings.

4. Run multi GPU distributed training:
    ```shell
    $ CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers you want to use" train.py
    python train.py -d 0 -f gfnet-b0-RGB3-DIN-T24-U8
    ```

## Evaluation
Run the evaluation by:
```shell
CUDA_VISIBLE_DEVICES="GPU IDs" python eval.py -d="Device ID" -f "config filename in configs dir" -e="epoch number, range, or pth"

CUDA_VISIBLE_DEVICES="GPU IDs" python eval.py  -d=0 -f gfnet-b0-RGB3-I-D-T24-U8 -e pretrained/GFNET-ADD-CEDICE-FRM-FFM-RGB3-IAIP-D_FocusN-T24-U8_mit_b0.pth
```
If you want to use multi GPUs please specify multiple Device IDs (0,1,2...).


## Acknowledgement

Our code is heavily based on: [RGBX_Semantic_Segmentation](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)
which was based on: [TorchSeg](https://github.com/ycszen/TorchSeg) and [SA-Gate](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch), thanks for their excellent work!