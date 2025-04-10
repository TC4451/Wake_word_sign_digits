<h1> Synthesis and Full Chip Static Timing Analysis </h1>

We want to synthesize the BNN ( or a binarize layer ) and then perform Static Timing.

Moving the Model from FP32 in software to a gate level implementation should improve speed from milliseconds to nanosecond.

IIC-OSIC-TOOLS is an all-in-one Docker container for open-source-based integrated circuit designs for analog and digital circuit flows. This collection of tools is curated by the Institute for Integrated Circuits and Quantum Computing (IICQC), Johannes Kepler University (JKU).

[Location of tools](https://github.com/iic-jku/IIC-OSIC-TOOLS)

We will use the tool yosys for synthesize and opensta for static timing analsys

<h2>FLOW</h2>

Install docker for Windows 

Install WSL (or similar)

git clone --depth=1 https://github.com/iic-jku/iic-osic-tools.git

./start_shell.sh

<h2>Yosys<h2>

Read verilog

Read in sdk lib file

Link

Compile

Output gate level netlist


<h2>Opensta</h2>

Read gate level netlist

read sdk lib file

create clocks

link

check_timing

report_paths

