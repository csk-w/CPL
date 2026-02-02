# CPL

# Data Preparation

VeRI-776 ; CityFlow-ReID ; VRIC ; MSMT17 

# Equipment

NVIDIA RTX 3090 GPU * 2

Ubuntu 20.04

# Command

```shell
CUDA_VISIBLE_DEVICES=0,1 python CPL/main.py  --dataset veri --height 224 --width 224 --num-clusters 1000 --iters 400 --num-epochs 80  --arch resnet50 --resnet-pretrained V1 --pooling-type gem --hdc-outlier --hdc-centroids --hdc-init k-means++2 --cm-mode hd_camera --loss-with-camera --root-dir $HOME/Dataset --log-dir logs/log   --proto_temp 0.08   --lambda_cons 1.0  --ema_alpha 0.999  --beta 0.2   --instance_temp 0.08   --beta_ramp_epochs 20  --clusterer hcf_gnn  --k0_factor 2.0 --knn_m 10
```


