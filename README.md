# ChargePal UR-Pilot

Python package to have high-level control of the URx robot arm

## Installation
There are several possibilities to install python packages. Mostly depends on how do you want to use them.
In the following two approaches are listed.

### To use as a module
```shell
pip install git+https://github.com/DFKI-ChargePal/chargepal_ur_pilot.git
```

### To use the scripts
```shell
git clone https://github.com/DFKI-ChargePal/chargepal_ur_pilot.git
cd ur_pilot
pip install -e .

# Clone configuration repository into the ur_pilot repository
git clone https://github.com/DFKI-ChargePal/chargepal_configuration.git config
# Check out a proper configuration branch
cd config
git switch -b xxx/xxx
```

## Getting started

The folder [scripts](./scripts) contains example code to control the arm.
