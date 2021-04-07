# LoFTR: Detector-Free Local Feature Matching with Transformers
### [Project Page](https://zju3dv.github.io/loftr) | [Paper](https://arxiv.org/pdf/2104.00680.pdf)
<br/>

> LoFTR: Detector-Free Local Feature Matching with Transformers  
> [Jiaming Sun](https://jiamingsun.ml)<sup>\*</sup>, [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>, [Yu'ang Wang](https://github.com/angshine)<sup>\*</sup>, [Hujun Bao](http://www.cad.zju.edu.cn/bao/), [Xiaowei Zhou](http://www.cad.zju.edu.cn/home/xzhou/)  
> CVPR 2021

![demo_vid](assets/loftr-github-demo.gif)


## Installation
```shell
# For full pytorch-lightning trainer features
conda env create -f environment.yaml
conda activate loftr

# For the LoFTR matcher only
pip install torch einops yacs kornia
```

We provide the [download link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) to 
  - the scannet-1500-testset (~1GB).
  - the megadepth-1500-testset (~600MB).
  - 4 pretrained models of indoor-ds, indoor-ot, outdoor-ds and outdoor-ot (each ~45MB).

By now, the LoFTR-DS model is ready to go!

<details>
  <summary>[Requirements for LoFTR-OT]</summary>

  We use the code from [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) for optimal transport. However, we can't provide the code directly due to its LICENSE. We recommend downloading it instead. 

  ```shell
  cd src/loftr/utils  
  wget https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/superglue.py 
  ```
</details>


## Run the code

### Match image pairs with LoFTR

<details>
  <summary>[code snippets]</summary>

  ```python
  from src.loftr import LoFTR, default_cfg

  # Initialize LoFTR
  matcher = LoFTR(config=default_cfg)
  matcher.load_state_dict(torch.load("weights/indoor_ds.ckpt")['state_dict'])
  matcher = matcher.eval().cuda()

  # Inference
  with torch.no_grad():
      matcher(batch)    # batch = {'image0': img0, 'image1': img1}
      mkpts0 = batch['mkpts0_f'].cpu().numpy()
      mkpts1 = batch['mkpts1_f'].cpu().numpy()
  ```

</details>

An example is in the `notebooks/demo_single_pair.ipynb`.

### Reproduce the testing results with pytorch-lightning

```shell
conda activate loftr
# with shell script
bash ./scripts/reproduce_test/indoor_ds.sh

# or
python test.py configs/data/scannet_test_1500.py configs/loftr/loftr_ds.py --ckpt_path weights/indoor_ds.ckpt --profiler_name inference --gpus=1 --accelerator="ddp"
```

For visualizing the dump results, please refer to `notebooks/visualize_dump_results.ipynb`.

### Reproduce the training phase with pytorch-lightning

The code is coming soon, stay tuned!

<br/>


## Code release ETA
The entire codebase for data pre-processing, training and validation is under major refactoring and will be released around June.
Please subscribe to [this discussion thread](https://github.com/zju3dv/LoFTR/discussions/2) if you wish to be notified of the code release.
In the meanwhile, discussions about the paper are welcomed in the [discussion panel](https://github.com/zju3dv/LoFTR/discussions).


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{sun2021loftr,
  title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal={CVPR},
  year={2021}
}
```

<!-- ## Acknowledgment

This repo is built based on the Mask R-CNN implementation from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), and we also use the pretrained Stereo R-CNN weight from [here](https://drive.google.com/file/d/1rZ5AsMms7-oO-VfoNTAmBFOr8O2L0-xt/view?usp=sharing) for initialization. -->


## Copyright
This work is affiliated with ZJU-SenseTime Joint Lab of 3D Vision, and its intellectual property belongs to SenseTime Group Ltd.

```
Copyright SenseTime. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


