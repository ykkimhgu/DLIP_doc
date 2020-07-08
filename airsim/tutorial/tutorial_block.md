# T\#2

You can set-up your own Unreal Environment with [these instructions](https://github.com/Microsoft/AirSim/#how-to-get-it).

For this tutorial, we will use a simple block environment.

Blocks environment is available in repo in folder `Unreal/Environments/Blocks` and is designed to be lightweight in size. That means its very basic but fast.

Here are quick steps to get Blocks environment up and running:

## Windows

1. Make sure you have [installed Unreal and built AirSim](../setup/build-on-windows.md).
2. Make sure you have `uproject` files are associated with Unreal engine
   * After installation of Unreal, Restart computer
   * Open `Epic Game Launcher` -&gt; `Unreal Engine` -&gt; `Unreal Engine 4.24.x`

     ![epic1](../../.gitbook/assets/epic1.jpg)

   * Associate `uproject` with Unreal engine

     ![epic1](../../.gitbook/assets/epic2.jpg)
3. Open `Command Prompt for VS 2019` and navigate to folder `AirSim\Unreal\Environments\Blocks` and run `update_from_git.bat`. Or you can run `update_from_git.bat` by double clicking the file in File Explorer
4. With File Explorer,  double click on generated .sln file to open in Visual Studio 2019 or newer.
5. Make sure `Blocks` project is the startup project. Click the right mouse button and select `Startup project`.

   ![block1](../../.gitbook/assets/block1.jpg)

6. Build configuration is set to `DebugGame_Editor` and `Win64`. Hit F5 to run.

   ![block2](../../.gitbook/assets/block2.jpg)

7. Press the Play button in Unreal Editor and you will see something like in below video. Also see [how to use AirSim](https://github.com/Microsoft/AirSim/#how-to-use-it).

   [![Blocks Demo Video](../../.gitbook/assets/blocks_video.png)](https://www.youtube.com/watch?v=-r_QGaxMT4A)

### Changing Code and Rebuilding

For Windows, you can just change the code in Visual Studio, press F5 and re-run. There are few batch files available in folder `AirSim\Unreal\Environments\Blocks` that lets you sync code, clean etc.

## FAQ

#### I see warnings about like "\_BuitData" file is missing.

These are intermediate files and you can safely ignore it.

