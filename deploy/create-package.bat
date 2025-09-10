@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM K8s Assistant - Deployment Package Creation Script (Windows)
REM Package all necessary files into a compressed archive for native deployment

echo INFO: Creating K8s Assistant native deployment package...

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set DEPLOY_DIR=%SCRIPT_DIR%
set PACKAGE_NAME=k8s-assistant-native-%date:~-10,4%%date:~-5,2%%date:~-2,2%-%time:~0,2%%time:~3,2%%time:~6,2%
set PACKAGE_NAME=%PACKAGE_NAME: =0%
set PACKAGE_DIR=%DEPLOY_DIR%%PACKAGE_NAME%

echo Project root: %PROJECT_ROOT%
echo Deploy directory: %DEPLOY_DIR%
echo Package name: %PACKAGE_NAME%

REM Check required tools
echo INFO: Checking required tools...
where tar >nul 2>&1
if errorlevel 1 (
    echo ERROR: tar tool not found
    echo Please install Git for Windows or 7-Zip
    pause
    exit /b 1
)

REM Create package directory
echo INFO: Creating package directory...
if exist "%PACKAGE_DIR%" rmdir /s /q "%PACKAGE_DIR%"
mkdir "%PACKAGE_DIR%"

REM Copy backend files
echo INFO: Copying backend files...
mkdir "%PACKAGE_DIR%\backend"
xcopy "%PROJECT_ROOT%\backend\app" "%PACKAGE_DIR%\backend\app" /E /I /Q
copy "%PROJECT_ROOT%\backend\main.py" "%PACKAGE_DIR%\backend\"
copy "%PROJECT_ROOT%\backend\requirements.txt" "%PACKAGE_DIR%\backend\"

REM Copy frontend files
echo INFO: Copying frontend files...
mkdir "%PACKAGE_DIR%\frontend"
xcopy "%PROJECT_ROOT%\frontend\src" "%PACKAGE_DIR%\frontend\src" /E /I /Q
copy "%PROJECT_ROOT%\frontend\package.json" "%PACKAGE_DIR%\frontend\"
if exist "%PROJECT_ROOT%\frontend\package-lock.json" copy "%PROJECT_ROOT%\frontend\package-lock.json" "%PACKAGE_DIR%\frontend\"
copy "%PROJECT_ROOT%\frontend\vite.config.ts" "%PACKAGE_DIR%\frontend\"
copy "%PROJECT_ROOT%\frontend\tsconfig.json" "%PACKAGE_DIR%\frontend\"
copy "%PROJECT_ROOT%\frontend\index.html" "%PACKAGE_DIR%\frontend\"

REM Copy nginx files
echo INFO: Copying nginx files...
mkdir "%PACKAGE_DIR%\nginx"
xcopy "%PROJECT_ROOT%\nginx" "%PACKAGE_DIR%\nginx" /E /I /Q

REM Copy data processing module
echo INFO: Copying data processing module...
mkdir "%PACKAGE_DIR%\data_processing"
xcopy "%PROJECT_ROOT%\data_processing" "%PACKAGE_DIR%\data_processing" /E /I /Q

REM Copy documentation
echo INFO: Copying documentation...
mkdir "%PACKAGE_DIR%\docs"
xcopy "%PROJECT_ROOT%\docs" "%PACKAGE_DIR%\docs" /E /I /Q

REM Copy startup scripts
echo INFO: Copying startup scripts...
copy "%PROJECT_ROOT%\start-native.sh" "%PACKAGE_DIR%\"
copy "%PROJECT_ROOT%\setup-native.sh" "%PACKAGE_DIR%\"
copy "%PROJECT_ROOT%\stop-native.sh" "%PACKAGE_DIR%\"
copy "%PROJECT_ROOT%\start_milvus.py" "%PACKAGE_DIR%\"

REM Copy environment configuration template
echo INFO: Copying environment configuration template...
copy "%PROJECT_ROOT%\.env.example" "%PACKAGE_DIR%\"

REM Copy deployment documentation
echo INFO: Copying deployment documentation...
copy "%DEPLOY_DIR%\DEPLOY.md" "%PACKAGE_DIR%\"

REM Create version information
echo INFO: Creating version information...
(
    echo K8s Assistant Native Deployment Package
    echo Version: 1.0.0
    echo Build Date: %date% %time%
    echo Build Host: %COMPUTERNAME%
    echo Git Commit: unknown
) > "%PACKAGE_DIR%\VERSION"

REM Create compressed package
echo INFO: Creating compressed package...
cd /d "%DEPLOY_DIR%"
tar -czf "%PACKAGE_NAME%.tar.gz" "%PACKAGE_NAME%"

REM Calculate file size
for %%A in ("%PACKAGE_NAME%.tar.gz") do set PACKAGE_SIZE=%%~zA
set /a PACKAGE_SIZE_MB=%PACKAGE_SIZE%/1024/1024

REM Clean temporary directory
echo INFO: Cleaning temporary files...
rmdir /s /q "%PACKAGE_DIR%"

REM Display results
echo.
echo SUCCESS: Deployment package created successfully!
echo.
echo INFO: Package information:
echo    - File name: %PACKAGE_NAME%.tar.gz
echo    - Size: %PACKAGE_SIZE_MB% MB
echo    - Location: %DEPLOY_DIR%\%PACKAGE_NAME%.tar.gz
echo.
echo INFO: Deployment steps:
echo Step 1: Transfer %PACKAGE_NAME%.tar.gz to target machine (Linux/macOS)
echo Step 2: Extract: tar -xzf %PACKAGE_NAME%.tar.gz
echo Step 3: Enter directory: cd %PACKAGE_NAME%
echo Step 4: setup: setup-native.sh
echo Step 4: start: start-native.sh
echo.
echo INFO: For detailed instructions, see DEPLOY.md file in the package

pause
