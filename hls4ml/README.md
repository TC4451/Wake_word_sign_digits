<h1>HSL4ML</h1>

[HLS4ML](https://fastmachinelearning.org/hls4ml/)  is a Python package for machine learning inference in FPGAs.

We want to use this package to create a verilog model of the Wake Word BNN.

After we get the verilog we can synthesize it to a gate level netlist.

To get HLS2ML to work we had to download [vitis](https://www.xilinx.com/support/download.html)

After installation you need to source /tools/Xilinx/Vitis/2024.2/settings64.sh  to get vitis setup.

<h2>Setup HLS4ML</h2>

A tutorial for HLS4ML can be found [here](https://github.com/fastmachinelearning/hls4ml-tutorial/blob/main/part1_getting_started.ipynb)

Load model, setup input parameters, compile model, build model

config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=config, backend='Vitis', output_dir='model_1/hls4ml_prj', part='xcu250-figd2104-2L-e'
)
hls_model.compile()

hls_model.build(csim=False)

<h2>Results</h2>
