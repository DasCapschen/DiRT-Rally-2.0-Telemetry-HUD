# DiRT HUD

A dirty hack of MangoHUD (which I forked from) to display Dirt Rally 2.0 Telemetry Data ingame.  

I am very sorry to the creators of MangoHUD for butchering their code, but also very thankful that they have created this amazing project!

# Installation
First clone this repository and cd into it:

```
git clone --recurse-submodules https://github.com/DasCapschen/DiRTHUD.git
cd DiRTHUD
```

Then simply run the following command:

`./build.sh install`

This will build and copy `libDiRTHud.so` & `libDiRTHud32.so` to `$HOME/.local/share/DiRTHud`, as well as copying the required Vulkan layer configuration files.

---

If you are running an Ubuntu-like distrobution, Fedora, or Arch, the build script will automatically detect and prompt you to install missing build dependencies. If you run in to any issues with this please report them!

# Normal usage

To enable the DirtHud Vulkan overlay layer, run :

`DIRTHUD=1 /path/to/my_vulkan_app`

Or alternatively, add `DIRTHUD=1` to your shell profile.

## Enable Telemetry
Find `My Documents/My Games/DiRT Rally 2.0/hardwaresettings/hardware_settings_config.xml` and open it in your favourite editor.  
Find `<udp enabled="false" extradata="0" ip="127.0.0.1" port="20777" delay="1" />` and change it to `<udp enabled="true" extradata="3" ip="127.0.0.1" port="20777" delay="1" />`  

### Note to Linux (Proton) Users
The file is located in `~/.local/share/Steam/steamapps/compatdata/690790/pfx/drive_c/users/steamuser/` and then the path as above.  
For plain Wine, look in `~/.wine/drive_c/users/<your username>/` followed by the path above.