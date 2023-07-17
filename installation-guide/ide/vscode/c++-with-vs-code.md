# C++ with VS Code

## C/C++ for Visual Studio Code

https://code.visualstudio.com/docs/languages/cpp

VS Code is an editor, NOT IDE. It does not have C++ compiler or debugger. You will need to install these tools or use those already installed on your computer.

For Window:

* GCC, MSVC
* MSVC is installed when Visual Studio Community is installed.

MacOS:

* GCC, Clang
* macOS users can get the [Clang](https://wikipedia.org/wiki/Clang) tools with [Xcode](https://developer.apple.com/xcode/)

### Steps

1. Install C++ Extension in VS Code
2. Install Compiler (GCC)

***

## Step 1: Install C++ Extension in VS Code

1. Open VS Code.
2. Select the Extensions view icon on the Activity bar or use the keyboard shortcut (Ctrl+Shift+X).
3. Search for `'C++'`.
4. Select **Install**.

![Search for c++ in the Extensions view](https://code.visualstudio.com/assets/docs/languages/cpp/search-cpp-extension.png)

After you install the extension, when you open or create a `*.cpp` file, you will have syntax highlighting (colorization), smart completions and hovers (IntelliSense), and error checking.

## Step 2: Install Compiler

Lets use GCC for this tutorial.

#### Check if you have a compiler installed

Make sure your compiler executable is in your platform path (`%PATH` on Windows, `$PATH` on Linux and macOS) so that the C/C++ extension can find it. You can check availability of your C++ tools by opening the Integrated Terminal (Ctrl+\`) in VS Code and trying to directly run the compiler.

Checking for the GCC compiler `g++`:

```
g++ --version
```

Checking for the Clang compiler `clang`:

```
clang --version
```

#### Install MinGW-x64 for GCC

Follow the **Installation** instructions on the [MSYS2 website](https://www.msys2.org/) to install Mingw-w64.

> Take care to run each required Start menu and `pacman` command.

Open MSYS2 with terminal for the UCRT64 environment

Install mingw-w64 GCC by

`pacman -S mingw-w64-ucrt-x86_64-gcc`

You will need to install the full Mingw-w64 toolchain to get the `gdb` debugger.

`pacman -S --needed base-devel mingw-w64-x86_64-toolchain`

Then, select `mingw-w64-x86_64-gbd`

![image-20230717172842570](https://c/Users/ykkim/AppData/Roaming/Typora/typora-user-images/image-20230717172842570.png)

#### Add the MinGW compiler to your path

Add the path to your Mingw-w64 `bin` folder to the Windows `PATH` environment variable by using the following steps:

1. In the Windows search bar, type 'settings (설정)' to open your Windows Settings.
2. Search for **Edit environment variables for your account**. **(계정의 환경 변수 편집)**
3. Choose the `Path` variable in your **User variables** and then select **Edit**.
4. Select **New** and add the Mingw-w64 destination folder path: `C:\msys64\mingw64\bin`.
5. Select **OK** to save the updated PATH. You will need to reopen any console windows for the new PATH location to be available.
6. ![image-20230717174320564](https://c/Users/ykkim/AppData/Roaming/Typora/typora-user-images/image-20230717174320564.png)

To check your MinGW installation, open a **new** Command Prompt and type:

```
gcc --version
```

#### Test Code

We'll create the simplest Hello World C++ program.

Create a folder called "HelloWorld" and open VS Code in that folder (`code .` opens VS Code in the current folder):

```
mkdir HelloWorld
cd HelloWorld
code .
```

Accept the [Workspace Trust](https://code.visualstudio.com/docs/editor/workspace-trust) dialog by selecting **Yes, I trust the authors**

> You can also open VS Code directly.

Now create a new file called `helloworld.cpp` with the **New File** button in the File Explorer or **File** > **New File** command.

![File Explorer New File button](https://code.visualstudio.com/assets/docs/languages/cpp/new-file.png)

![helloworld.cpp file](https://code.visualstudio.com/assets/docs/languages/cpp/hello-world-cpp.png)

Now paste in this source code:

```
#include <iostream>

int main()
{
    std::cout << "Hello World" << std::endl;
}
```

Now press Ctrl+S to save the file. You can also enable [Auto Save](https://code.visualstudio.com/docs/editor/codebasics#\_save-auto-save) to automatically save your file changes, by checking **Auto Save** in the main **File** menu.

**Build Hello World**

Now that we have a simple C++ program, let's build it. Select the **Terminal** > **Run Build Task** command (Ctrl+Shift+B) from the main menu.

![Run Build Task menu option](https://code.visualstudio.com/assets/docs/languages/cpp/run-build-task.png)

Choose GCC toolset MinGW: **C/C++: g++.exe build active file**.

![Select g++.exe task](https://code.visualstudio.com/assets/docs/languages/cpp/gpp-build-task-msys64.png)

This will compile `helloworld.cpp` and create an executable file called `helloworld.exe`, which will appear in the File Explorer.

![helloworld.exe in the File Explorer](https://code.visualstudio.com/assets/docs/languages/cpp/hello-world-exe.png)

**Run Hello World**

VS Code Integrated Terminal, run your program by typing ".\helloworld".

> You can also run by Ctrl+F5

![Run hello world in the VS Code Integrated Terminal](https://code.visualstudio.com/assets/docs/languages/cpp/run-hello-world.png)

If everything is set up correctly, you should see the output "Hello World".
