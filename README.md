[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GiorgioMorales/OscillationMaps/blob/master/Demo_OscillAI.ipynb)


# OscillAI: Simulating Neutrino Oscillation Maps using AI

## Installation

The following libraries have to be installed:
* [Git](https://git-scm.com/download/) 
* [Pytorch](https://pytorch.org/get-started/locally/)

To install the package, run `pip install -q git+https://github.com/GiorgioMorales/OscillationMaps` in the terminal.

You can also try the package on [Google Colab](https://colab.research.google.com/github/GiorgioMorales/OscillationMaps/blob/master/Demo_OscillAI.ipynb).


## Usage

### Train the models

OscillAI uses the class `MatterEffect` to produce the 9 oscillation maps estimating matter effect.
By initializing this class, we're loading and prepping the saved trained models and model statistics.

```python
from OscillationMaps.MatterEffect import MatterEffect
propagator = MatterEffect()
```

To get the actual maps given a set of oscillation parameters, we call the `get_maps` method:

**Parameters** (for now):

*   `osc_pars`: Batches of oscillation parameters in order: `[theta12, theta23, theta13, delta_cp, m21, m31]`

```python
# Define 7 sets of parameters
osc_pars_in = np.array([[2.392e+02,  2.955e+02,  2.183e+02,  2.128e+02,  6.523e-05, -5.502e-03],
                           [2.0650e+02,  2.6060e+02,  1.1860e+02,  2.8160e+01,  6.7460e-05, -4.0790e-03],
                           [1.2760e+02, 1.8440e+02, 5.6400e+01, 3.1870e+02, 6.7620e-05, 9.7840e-03],
                           [1.6420e+02, 1.1820e+02, 3.1630e+00, 2.2710e+02, 6.7560e-05, -9.8820e-03],
                           [5.5570e+01, 1.9820e+02, 2.6070e+02, 2.8350e+02, 2.2470e-05, 9.5280e-03],
                           [1.6100e+02, 2.5260e+02, 2.0360e+02, 4.5740e+01, 8.3220e-05, 8.0640e-03],
                           [3.0250e+01, 3.1800e+02, 2.8610e+02, 2.0900e+01, 2.0050e-05, 2.5740e-03]])
# Propagate
maps_out = propagator.get_maps(osc_pars=osc_pars_in)
```

To plot the results, we call the `plot_osc_maps` method:
```python
from OscillationMaps.utils import plot_osc_maps
for ii in range(maps_out.shape[0]):
    plot_osc_maps(maps_out[ii], title='Predicted Oscillation Maps: ' + str(ii + 1))
```

Finally, we can also simulate the oscillation maps in vacuum using the function `get_oscillation_maps_vacuum`. 
Note that these maps are calculated analytically and, thus, don't need to use any trained model.

```python
from OscillationMaps.VacuumMaps import get_oscillation_maps_vacuum
# Define parameters
osc_pars_in = [2.392e+02,  2.955e+02,  2.183e+02,  2.128e+02,  6.523e-05, -5.502e-03]
# Propagate
maps_out = get_oscillation_maps_vacuum(osc_pars=osc_pars_in)

plot_osc_maps(maps_out, title='Oscillation Maps in Vacuum')
```