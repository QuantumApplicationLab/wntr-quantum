# NetLoops Stability Tests

This folder contains inputs for all stability tests done in connection with the wntr-quantum package development. The main idea is to randomly vary intrinsic parameters of the `Net2Loops` network, *e.g.*, pipe width, pipe length, node demands and Hazen-Williams roughness coefficients, and compare the quantum NR hydraulics simulation results (*e.g.*, pressures and flow rates) with those obtained using the classical WNTR `EpanetSimulator`.

## Download the output folder

To download the folder containing the results of all runs and plot scripts, follow [this link](https://drive.google.com/file/d/1-ONsXO_WwINUDrOq-AYyQjnlX6SHRgA1/view?usp=sharing)

## Plotting the results

After downloading the zip file, you will find inside each folder, *e.g.*, `vqls`, `qubols`, and `hhl`, python scripts that can be used to generate plots of the stability results, namely, `correlation_flows_analysis_stability_tests.py` `correlation_pressure_analysis_stability_tests.py`

Before running them, remember to activate your python or conda environments with `wntr` and `wntr-quantum` installed and them do:

```bash
python correlation_pressure_analysis_stability_tests.py
```

Apart from saving the plots in `*.png` format, the corresponding plots will also pop-up in your screen.
