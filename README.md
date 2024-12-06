# OAK-D LITE TESTS FOR EMBEDDED NN
This repository is intended to be a "wrapper" in combination with the depthai library from Luxonis (openai) to use Neural Networks embedded in their AI-capable deivce, OAK-D Lite (Kickstarter Version)

The repo provide capabilities (for now) for the calibration of the camera RGB sensor and storing those values in the device EEPROM
### Set udev ruels on linux systems
Run these commands on bash 
```
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### To Do:
* Add calibration for stereo vision
* Add calibration for mono cameras (stereo but separate)
* Assess if that makes any difference in the stereo output