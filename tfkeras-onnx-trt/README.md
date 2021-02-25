# TensorRT : tfkeras.inception_ve -> onnx -> serialized engine/plan


## data

dogs_vs_cats_saved_model/ 

Created by training a tf.keras.applications.inception_v3 model in 
channels_first data format using the Kaggle Dogs vs. Cats dataset.

## environment

- Build Docker

```
  nvidia-docker build --t fuzzy-octo-memory:v1 -f Dockerfile .
```

- Launch Docker, mapping in this code

```
  nvidia-docker run --net=host -it \
        -v `pwd`:/workspace/fuzzy \
        -w /workspace/fuzzy \
        fuzzy-octo-memory:v1
```


## Create serialized engine

```
   $ ./model-to-onnx-to-trt.sh
```

Uses 
- tf2onnx for saved_model -> onnx, while ictily setting input shape for batch size 1.
- trtexec for onnx -> trt


## inference in PYTHON

```
   $ ./inference-from-trt.py
  
     cat.0.jpg,cat,0.0000,dog,1.0000
     dog.0.jpg,cat,0.0002,dog,0.9998
     cat.1.jpg,cat,1.0000,dog,0.0000
     dog.1.jpg,cat,0.0001,dog,0.9999
```

## inference in CPP

   _assumes runnnig as root. otherwise, update Dockerfile to change permissions of /opt/tensorrt_

1. copy files into /opt/tensorrt/samples

```
   $ cp /workspace/fuzzy/cpp-files/opt.tensorrt.samples.Makefile.config /opt/tensorrt/samples/Makefile.config
   $ cp -r /workspace/fuzzy/cpp-files/sampleMine /opt/tensorrt/samples
   $ mkdir /opt/tensorrt/data/mine
   $ cp /workspace/fuzzy/dogs_vs_cats_model.trt images/* /opt/tensorrt/data/mine
```


2. make

```
   $ cd /opt/tensorrt/samples/sampleMine
   $ make
```   

3. execute

```
   $ ../../bin/sample_mine
```
