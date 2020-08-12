---
description: A short tutorial on how to install Airsim on Window 10.
---

# Setup

> Read airsim docs  on how to build Airsim [https://microsoft.github.io/AirSim/build\_windows](https://microsoft.github.io/AirSim/build_windows) on widows  for more detailed information
>
> For Airsim v. and Unreal Engine &gt;=4.22 
>
> Updated 2020-6

## Requirements

* Window 10
* NVIDIA Graphic card & Driver
* Unreal engine &gt;=4.22
* Visual Studio 2019 \(English version\)
* Git for windows

## Install On Windows <a id="on-windows"></a>

### Install Unreal Engine <a id="install-unreal-engine"></a>

1. ​[Download](https://www.unrealengine.com/download) the Epic Games Launcher. While the Unreal Engine is open source and free to download, registration is still required.

   ​​​‌

2. Run the Epic Games Launcher, open the `Library` tab on the left pane. Click on the `Add Versions` which should show the option to download **Unreal 4.24** as shown below. If you have multiple versions of Unreal installed then **make sure 4.24 is set to** **`current`** by clicking down arrow next to the Launch button for the version.

   **Note**: AirSim also works with UE &gt;= 4.22, however, we recommend you update to 4.24.

![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2Fb98bb2fcd3c88380c2b3f9da53a85db5f4b4b10f.jpg?alt=media)

​‌

![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2F0aa9b007473f7411472a76b0360d56c756a9cf38.jpg?alt=media)

### Install Git for Windows <a id="install-git-for-windows"></a>

* ​[Download](https://git-scm.com/) Git for Windows \(lastes source, release 2.27.0\)
* Use default settings to install

![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2F31be58c30532533eefac7da6ca8e31d795d3b847.jpg?alt=media)

### Install Visual Studio <a id="install-visual-studio"></a>

* Install Visual Studio 2019.

  > **Note**: Use **English** for the VS2019 language. If you choose **Korean**, you may need to convert 'UTF Encoding' is some AirSim source files.

* **Make sure** to select **Desktop Development with C++** and **Windows 10 SDK 10.0.18362** \(should be selected by default\) while installing VS 2019.

![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2F2f96a5b58c61231e6399e6b95a7e612c7d99e4bc.jpg?alt=media)

![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2Fca9578365549c3576af3f7f46d325019eb978d5d.jpg?alt=media)

### Download AirSim <a id="download-airsim"></a>

* Start `Git Bash` or `Git GUI`
* Move to the target directory to install AirSim. Use `cd` command to change directory
* Clone the AirSim repo by typing

```text
git clone https://github.com/Microsoft/AirSim.git
```

![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2F9af9c754644d72b14a9346fcb0c1f3a2491f02df.jpg?alt=media)

### Build AirSim <a id="build-airsim"></a>

* Open `Command Prompt for VS2019`
* Move to the installed AirSim directory by `cd AirSim`.
* Run `build.cmd` from the command line. This will create ready to use plugin bits in the `Unreal\Plugins` folder that can be dropped into any Unreal project.

> **Build error**: If the build error message is related to UTF decoding Go to FAQ \#1

![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2F03cfbd6fc4a97e718c471a515e6c9549ae524619.jpg?alt=media)

### Build Unreal Project <a id="build-unreal-project"></a>

Finally, you will need an Unreal project that hosts the environment for your vehicles. AirSim comes with a built-in "Blocks Environment" which you can use, or you can create your own.‌

Please see [setting up Built-in Block Environment](https://app.gitbook.com/@ykkim/s/wiki/~/drafts/-MEYPohdVfVbg_mME8yG/simulator/airsim/tutorial/tutorial_block).‌

Also, see [setting up Unreal Environment](https://microsoft.github.io/AirSim/unreal_proj/) for more detail.‌

## Troubleshooting <a id="troubleshooting"></a>

### Building Error message of Error C2220 <a id="building-error-message-of-error-c2220"></a>

The initial setting for VS2019 gives the warning as the compile error. This error has to do with some sources have some characters encoded which shows warning by the compiler.‌

* Open the header/source files that gives encoding warning with VS2019
* **Save As** with **Save with Encoding**: Encoding **UTF-8 with signature**

​![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2F0372e208fbae13123284b0cd92ebffd107b31253.jpg?alt=media)​![](https://gblobscdn.gitbook.com/assets%2F-MAwtzMy_pbrChIExFtN%2Fsync%2Fc8810f140df0e6756949dd99370cf0c9a146d3c8.jpg?alt=media)‌

### Clashes of VS 2017 and VS2019. How to force Unreal to use Visual Studio 2019 <a id="clashes-of-vs-2017-and-vs-2019-how-to-force-unreal-to-use-visual-studio-2019"></a>

If different versions of Visual Studio are installed. such as VS2017 and VS 2019, Unreal may be associated with VS2017. Building an AirSim project in VS2019 with Unreal associated with VS2017 can give build error.‌

Need to Change the Unreal configuration to associate with VS2019‌

* Open `C:\Users$user\AppData\Roaming\Unreal Engine\UnrealBuildTool\BuildConfiguration.xml`
* Modify as following

  ```text
  <?xml version="1.0" encoding="utf-8" ?><Configuration xmlns="https://www.unrealengine.com/BuildConfiguration"><BuildConfiguration>    </BuildConfiguration>    <VCProjectFileGenerator>        <Version>VisualStudio2019</Version>    </VCProjectFileGenerator>​    <WindowsPlatform>        <Compiler>VisualStudio2019</Compiler>    </WindowsPlatform></Configuration>
  ```

​





