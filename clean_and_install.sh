rm -rf build
rm -rf dist
rm -rf causal_conv1d.egg-info
rm -rf csrc/*.hip
rm -rf csrc/*_hip.h
rm -rf csrc/*_hip.cpp


pip3 uninstall causal_conv1d -y

python3 setup.py install
