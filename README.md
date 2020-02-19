# DiRT HUD

A dirty hack of MangoHUD (which I forked form) to display Dirt Rally 2.0 Telemetry Data ingame.  

I am very sorry to the creators of MangoHUD for butchering their code, but also very thankful that they have created this amazing project!

# Installation
First clone this repository and cd into it:

```
git clone --recurse-submodules https://github.com/flightlessmango/MangoHud.git
cd MangoHud
```

Then simply run the following command:

`./build.sh install`

This will build and copy `libMangoHud.so` & `libMangoHud32.so` to `$HOME/.local/share/MangoHud`, as well as copying the required Vulkan layer configuration files.

---

If you are running an Ubuntu-like distrobution, Fedora, or Arch, the build script will automatically detect and prompt you to install missing build dependencies. If you run in to any issues with this please report them!

# Normal usage

To enable the DirtHud Vulkan overlay layer, run :

`DIRTHUD=1 /path/to/my_vulkan_app`

Or alternatively, add `DIRTHUD=1` to your shell profile.