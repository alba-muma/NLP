@echo off

:: Variables del entorno
set VENV_NAME=llama_env

:: Crear el entorno virtual
echo Creando entorno virtual "%VENV_NAME%"...
python -m venv %VENV_NAME%
if %errorlevel% neq 0 (
    echo Error al crear el entorno virtual.
    exit /b 1
)

:: Activar el entorno virtual
echo Activando el entorno virtual...
call %VENV_NAME%\Scripts\activate

:: Instalar las dependencias
echo Instalando las dependencias...
pip install -r requirements.txt


