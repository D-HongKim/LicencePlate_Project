import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

onnx_model = onnx.load("./alignment_best.onnx")
tf_rep = prepare(onnx_model)

tf_model_path = 're_alignment'
tf_rep.export_graph(tf_model_path)
print(".pb model converted successfully.")
input_nodes = tf_rep.inputs
output_nodes = tf_rep.outputs
print("The names of the input nodes are: {}".format(input_nodes))
print("The names of the output nodes are: {}".format(output_nodes))

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

tflite_model = converter.convert()

tflite_model_path = './re_alignment/alignment_v2.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(".tflite model converted successfully.")
