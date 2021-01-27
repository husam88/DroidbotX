# DroidbotX
Fork of droidbot (https://github.com/honeynet/droidbot) that adds gym environment to Droidbot.

## Prerequisite
1. `Python` (both 2 and 3 are supported)
2. `Java`
3. `Android SDK`
4. Add `platform_tools` directory in Android SDK to `PATH`
5. (Optional) `OpenCV-Python` if you want to run DroidBot in cv mode.

## How to install
Clone this repo and intall with `pip`:

```shell
git clone
cd droidbot/
pip install -e .
```
## Other packages needed
1. Gym (https://github.com/openai/gym for installation instructions)
2. Stable-Baselines (https://github.com/hill-a/stable-baselines for installation instructions)
3. TensorFlow
4. pip install opencv-python

## Usage
1. Run Emulator or connect to real Android device using adb.
                               *If you are using Android emulator. Install Android Studio, select the Android Emulator component in the SDK Tools tab of the SDK Manager.
                               *If you are using Android real device or multiple devices, you may need to use `-d <device_serial>` to specify the target device. The easiest way to determine a device's serial number is calling adb devices.
2. Add App Under test to APKs directory.
3. From terminal and droidbot directory run the "start_q_learning.py" script. Example usage: 

`python3 start_q_learning.py -a App_Under_Test.apk -o /my_output_dir/ -is_emulator -policy gym`

4. In `my_output_dic directory` find `index.html` to view the UTG model.


