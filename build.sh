rm -rf build
rm -rf dist
rm -rf causal_conv1d.egg-info
rm -rf csrc/*.hip
rm -rf csrc/*_hip.h
rm -rf csrc/*_hip.cpp


pip3 uninstall causal_conv1d -y

HIP_ARCHITECTURES=${HIP_ARCHITECTURES:="gfx936;gfx938"} \
python3 setup.py bdist_wheel
