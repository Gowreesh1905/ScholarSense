@echo off
rem Shrinks Docker Desktop's virtual disk after its contents were deleted.
rem Right-click this file -> "Run as administrator". Close Docker Desktop first.

net session >nul 2>&1
if errorlevel 1 (
  echo This needs administrator rights: right-click the file and choose "Run as administrator".
  pause
  exit /b 1
)

echo Stopping Docker / WSL ...
wsl --shutdown

set VHDX=C:\Users\ASUS\AppData\Local\Docker\wsl\disk\docker_data.vhdx
for %%A in ("%VHDX%") do echo Size before: %%~zA bytes

(
  echo select vdisk file="%VHDX%"
  echo attach vdisk readonly
  echo compact vdisk
  echo detach vdisk
) > "%TEMP%\compact_docker.txt"

echo Compacting (takes a minute or two) ...
diskpart /s "%TEMP%\compact_docker.txt"
del "%TEMP%\compact_docker.txt"

for %%A in ("%VHDX%") do echo Size after:  %%~zA bytes
echo Done.
pause
