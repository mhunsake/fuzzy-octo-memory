python -m tf2onnx.convert --saved-model ./dogs_vs_cats_saved_model --opset 13 --output dogs_vs_cats_model.onnx --inputs inception_v3_input:0[1,3,299,299]
trtexec --onnx=./dogs_vs_cats_model.onnx --saveEngine=dogs_vs_cats_model.trt --buildOnly
