## Openpose to 3d-pose-baseline

This is the up-to-date code for the paper below and extends it to use Openpose outputs.

Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little.
_A simple yet effective baseline for 3d human pose estimation._
In ICCV, 2017. https://arxiv.org/pdf/1705.03098.pdf.

The code in this repository was mostly written by
[Julieta Martinez](https://github.com/una-dinosauria),
[Rayat Hossain](https://github.com/rayat137),
[Javier Romero](https://github.com/libicocco) and [S. Arash Hosseini](https://github.com/ArashHosseini)

This repository serves as a bridge between [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 
2d keypoint outputs and the network proposed in [the original repository](https://github.com/una-dinosauria/3d-pose-baseline) 


### Dependencies

* Python 3.9
* [cdflib](https://github.com/MAVENSDC/cdflib)
* [tensorflow](https://www.tensorflow.org/) 1.15

### Implementation

1.Get the data

Go to http://vision.imar.ro/human3.6m/, log in, and download the `D3 Positions` and `D2 Positions` files for subjects `[5, 6, 7, 8, 9, 11]`,
and put them under the folder `data/h36m`. 
Uncompress all the data with the tgz format using

```bash
cd data/h36m/
for file in *.tgz; do tar -xvzf $file; done
```

Finally, download the `code-v1.2.zip` file, unzip it, and copy the `metadata.xml` file under `data/h36m/`

Now, your data directory should look like this:

```bash
data/
  └── h36m/
    ├── metadata.xml
    ├── S11/
    ├── S5/
    ├── S6/
    ├── S7/
    ├── S8/
    └── S9/

```

There is one little fix we need to run for the data to have consistent names:

```bash
mv h36m/S1/MyPoseFeatures/D3_Positions/TakingPhoto.cdf \
   h36m/S1/MyPoseFeatures/D3_Positions/Photo.cdf

mv h36m/S1/MyPoseFeatures/D3_Positions/TakingPhoto\ 1.cdf \
   h36m/S1/MyPoseFeatures/D3_Positions/Photo\ 1.cdf

mv h36m/S1/MyPoseFeatures/D3_Positions/WalkingDog.cdf \
   h36m/S1/MyPoseFeatures/D3_Positions/WalkDog.cdf

mv h36m/S1/MyPoseFeatures/D3_Positions/WalkingDog\ 1.cdf \
   h36m/S1/MyPoseFeatures/D3_Positions/WalkDog\ 1.cdf
```

### Quick demo

For a quick demo, you can train for one epoch and visualize the results. To train, run

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --epochs 1`

This should take about <5 minutes to complete on a GTX 1080, and give you around 56 mm of error on the test set.

Now, to visualize the results, run

`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --epochs 1 --sample --load 24371`


### Training

To train a model with clean 2d detections, run:

<!-- `python src/predict_3dpose.py --camera_frame --residual` -->
`python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise`

### Testing

To test the performance of the model using the extracted 2D keypoints of a video, run:

`python src/openpose_to_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --epochs 1 --load 97484 --openpose_json_dir ballet`

### Citing

```
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```

### Other implementations

* [Pytorch](https://github.com/weigq/3d_pose_baseline_pytorch) by [@weigq](https://github.com/weigq)
* [MXNet/Gluon](https://github.com/lck1201/simple-effective-3Dpose-baseline) by [@lck1201](https://github.com/lck1201)

### Extensions

* [@ArashHosseini](https://github.com/ArashHosseini) maintains [a fork](https://github.com/ArashHosseini/3d-pose-baseline) for estimating 3d human poses using the 2d poses estimated by either [OpenPose](https://github.com/ArashHosseini/openpose) or [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) as input.

### License
MIT
