@echo off
chcp 936 >nul
rem ============================================================
rem  音乐驱动 3D 舞蹈生成 · 本地工作台
rem
rem  双击本文件即可启动。启动后浏览器会自动打开一个页面，
rem  在页面上选音乐、点开始就行。
rem
rem  关掉浏览器页面，工作台会自己退出，不用去任务管理器杀进程。
rem  如果还有任务在跑，它会先把任务跑完再退出。
rem ============================================================
cd /d "%~dp0"
title 音乐舞蹈工作台

if exist "tools\serve.py" goto :haspy
echo [错误] 没在当前目录找到 tools\serve.py
echo        请把本文件放在「跳舞模型」文件夹里再双击。
echo.
pause
exit /b 1

:haspy
set "PY="
for %%i in (python.exe) do set "PY=%%~$PATH:i"
if defined PY goto :verify
goto :findlocal

:verify
rem 排除 Windows 商店那个假 python.exe（它会直接跳转商店）
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 set "PY="
if not defined PY goto :findlocal
goto :run

:findlocal
rem 退而求其次：Windows 自带的 py 启动器（装了官方 Python 就有）
set "PY=py"
py -c "import sys" >nul 2>&1
if errorlevel 1 goto :findlocal2
goto :run

:findlocal2
rem 再不行：本机已知的几个 Python 安装位置
set "PY="
if exist "C:\Users\dapanji\AppData\Local\Programs\Python\Python310\python.exe" set "PY=C:\Users\dapanji\AppData\Local\Programs\Python\Python310\python.exe"
if not defined PY if exist "C:\Users\dapanji\.workbuddy-ai\binaries\python\versions\3.13.12\python.exe" set "PY=C:\Users\dapanji\.workbuddy-ai\binaries\python\versions\3.13.12\python.exe"
if not defined PY goto :nopy
goto :run

:nopy
echo [错误] 没找到可用的 python.exe
echo        工作台本身只用标准库，装个 Python 3.9 以上版本就行。
echo.
pause
exit /b 1

:run
echo ============================================================
echo   正在启动工作台……
echo   浏览器会自动打开，页面地址是 http://127.0.0.1:18066/
echo   用完之后直接关掉浏览器页面，工作台就会自己退出。
echo ============================================================
echo.
"%PY%" "tools\serve.py"
set "RC=%errorlevel%"
echo.
if not "%RC%"=="0" goto :failed
echo [完成] 工作台已停止，可以关掉这个窗口了。
timeout /t 5 >nul
exit /b 0

:failed
echo [失败] 工作台异常退出，退出码 %RC%
echo        把上面的报错内容截图发出来看看。
echo.
pause
exit /b %RC%
