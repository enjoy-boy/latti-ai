#!/bin/bash
set -e

# 1. 运行 Python 端的计算图生成单元测试
echo "==== Running Python Graph Generation Test ===="
python3 -m unittest inference.unittests.test_gen_layers.TestLayerExport.test_softmax -v
python3 -m unittest inference.unittests.test_gen_layers.TestLayerExport.test_softmax_range_sweep -v

# 2. 编译 C++ 测试工程
echo "==== Compiling C++ Unit Tests ===="
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make test_fhe_layers_hetero -j$(nproc)

# 3. 运行 C++ 算子推理单元测试 (针对 Softmax 单点测试)
echo "==== Running C++ Softmax Single Test ===="
./inference/unittests/test_fhe_layers_hetero "*softmax_layer_single*"

# 4. 运行 C++ 算子推理单元测试 (针对 Softmax 区间扫描测试)
# 还没有实现图的推理
echo "==== Running C++ Softmax Range Sweep Test ===="
./inference/unittests/test_fhe_layers_hetero "*softmax_layer*"

echo "==== All Tests Completed Successfully ===="