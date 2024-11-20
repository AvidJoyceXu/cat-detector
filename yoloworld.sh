pwd
cd YOLO-World
# python deploy/onnx_demo.py ../ckpt/yolow-l.onnx ../multianimals "cat"
# python deploy/onnx_demo.py ../ckpt/yolow-l.onnx ../benchmarks_v2/train/images "cat"
# python deploy/onnx_demo.py ../ckpt/yolow-l.onnx ../provided-examples/images "cat"
python deploy/onnx_demo.py ../ckpt/yolow-l.onnx ../benchmarks_v2/valid/images "cat"