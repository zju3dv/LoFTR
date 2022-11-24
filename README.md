# LoFTR: Detector-Free Local Feature Matching with Transformers
### [Project Page](https://zju3dv.github.io/loftr) | [Paper](https://arxiv.org/pdf/2104.00680.pdf)
<br/>

> LoFTR: Detector-Free Local Feature Matching with Transformers  
> [Jiaming Sun](https://jiamingsun.ml)<sup>\*</sup>, [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>, [Yu'ang Wang](https://github.com/angshine)<sup>\*</sup>, [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/), [Xiaowei Zhou](http://www.cad.zju.edu.cn/home/xzhou/)  
> CVPR 2021

![demo_vid](assets/loftr-github-demo.gif)

## TODO List and ETA
- [x] Inference code and pretrained models (DS and OT) (2021-4-7)
- [x] Code for reproducing the test-set results (2021-4-7)
- [x] Webcam demo to reproduce the result shown in the GIF above (2021-4-13)
- [x] Training code and training data preparation (expected 2021-6-10)

Discussions about the paper are welcomed in the [discussion panel](https://github.com/zju3dv/LoFTR/discussions).

:thinking: **FAQ**

1. Undistorted images from D2Net are not available anymore.  
   For a temporal alternative, please use the undistorted images provided by the MegaDepth_v1 (should be downloaded along with the required depth files). We numerically compared these images and only found very subtle difference.

:triangular_flag_on_post: **Updates**
- Check out [QuadTreeAttention](https://github.com/Tangshitao/QuadTreeAttention), a new attention machanism that improves the efficiency and performance of LoFTR with less demanding GPU requirements for training.
- :white_check_mark: Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Kornia-LoFTR)
## Colab demo
Want to run LoFTR with custom image pairs without configuring your own GPU environment? Try the Colab demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BgNIOjFHauFoNB95LGesHBIjioX74USW?usp=sharing)

## Using from kornia

LoFTR is integrated into [kornia](https://github.com/kornia/kornia) library since version 0.5.11.

```
pip install kornia
```

Then you can import it as 
```python3
from kornia.feature import LoFTR
```

See tutorial on using LoFTR from kornia [here](https://kornia-tutorials.readthedocs.io/en/latest/image_matching.html).


## Installation
```shell
# For full pytorch-lightning trainer features (recommended)
conda env create -f environment.yaml
conda activate loftr

# For the LoFTR matcher only
pip install torch einops yacs kornia
```

We provide the [download link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) to 
  - the scannet-1500-testset (~1GB).
  - the megadepth-1500-testset (~600MB).
  - 4 pretrained models of indoor-ds, indoor-ot, outdoor-ds and outdoor-ot (each ~45MB).

By now, the environment is all set and the LoFTR-DS model is ready to go! 
If you want to run LoFTR-OT, some extra steps are needed:

<details>
  <summary>[Requirements for LoFTR-OT]</summary>

  We use the code from [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) for optimal transport. However, we can't provide the code directly due its strict LICENSE requirements. We recommend downloading it with the following command instead. 

  ```shell
  cd src/loftr/utils  
  wget https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/superglue.py 
  ```
</details>


## Run LoFTR demos

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

An example is given in `notebooks/demo_single_pair.ipynb`.

### Online demo
Run the online demo with a webcam or video to reproduce the result shown in the GIF above.
```bash
cd demo
./run_demo.sh
```
<details>
  <summary>[run_demo.sh]</summary>

  ```bash
  #!/bin/bash
  set -e
  # set -x

  if [ ! -f utils.py ]; then
      echo "Downloading utils.py from the SuperGlue repo."
      echo "We cannot provide this file directly due to its strict licence."
      wget https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/utils.py
  fi

  # Use webcam 0 as input source. 
  input=0
  # or use a pre-recorded video given the path.
  # input=/home/sunjiaming/Downloads/scannet_test/$scene_name.mp4

  # Toggle indoor/outdoor model here.
  model_ckpt=../weights/indoor_ds.ckpt
  # model_ckpt=../weights/outdoor_ds.ckpt

  # Optionally assign the GPU ID.
  # export CUDA_VISIBLE_DEVICES=0

  echo "Running LoFTR demo.."
  eval "$(conda shell.bash hook)"
  conda activate loftr
  python demo_loftr.py --weight $model_ckpt --input $input
  # To save the input video and output match visualizations.
  # python demo_loftr.py --weight $model_ckpt --input $input --save_video --save_input

  # Running on remote GPU servers with no GUI.
  # Save images first.
  # python demo_loftr.py --weight $model_ckpt --input $input --no_display --output_dir="./demo_images/"
  # Then convert them to a video.
  # ffmpeg -framerate 15 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4

  ```
</details>

### Reproduce the testing results with pytorch-lightning
You need to setup the testing subsets of ScanNet and MegaDepth first. We create symlinks from the previously downloaded datasets to `data/{{dataset}}/test`.

```shell
# set up symlinks
ln -s /path/to/scannet-1500-testset/* /path/to/LoFTR/data/scannet/test
ln -s /path/to/megadepth-1500-testset/* /path/to/LoFTR/data/megadepth/test
```

```shell
conda activate loftr
# with shell script
bash ./scripts/reproduce_test/indoor_ds.sh

# or
python test.py configs/data/scannet_test_1500.py configs/loftr/loftr_ds.py --ckpt_path weights/indoor_ds.ckpt --profiler_name inference --gpus=1 --accelerator="ddp"
```

For visualizing the results, please refer to `notebooks/visualize_dump_results.ipynb`.

<br/>


<!-- ### Image pair info for training on ScanNet
You can download the data at [here](https://drive.google.com/file/d/1fC2BezUSsSQy7_H65A0ZfrYK0RB3TXXj/view?usp=sharing).

<details>
  <summary>[data format]</summary>

```python
In [14]: npz_path = './cfg_1513_-1_0.2_0.8_0.15/scene_data/train/scene0000_01.npz'

In [15]: data = np.load(npz_path)

In [16]: data['name']
Out[16]:
array([[   0,    1,  276,  567],
       [   0,    1, 1147, 1170],
       [   0,    1,  541, 5757],
       ...,
       [   0,    1, 5366, 5393],
       [   0,    1, 2607, 5278],
       [   0,    1,  736, 5844]], dtype=uint16)

In [17]: data['score']
Out[17]: array([0.2903, 0.7715, 0.5986, ..., 0.7227, 0.5527, 0.4148], dtype=float16)

In [18]: len(data['name'])
Out[18]: 1684276

In [19]: len(data['score'])
Out[19]: 1684276
```
`data['name']` is the image pair info, organized as [`scene_id`, `seq_id`, `image0_id`, `image1_id`].

`data['score']` is the overlapping score defined in [SuperGlue](https://arxiv.org/pdf/1911.11763) (Page 12).
</details> -->


## Training
See [Training LoFTR](./docs/TRAINING.md) for more details.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{sun2021loftr,
  title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal={{CVPR}},
  year={2021}
}
```

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


