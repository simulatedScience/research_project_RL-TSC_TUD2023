@echo off
REM --- Variables ---
SETLOCAL ENABLEDELAYEDEXPANSION
SET FOLDER_PATH="data\output_data\tsc\sumo_presslight\sumo1x3_synth_uniform"
SET FILENAME_PREFIX="exp_new_"
SET PYTHON_SCRIPT="run_tests.py"

REM --- Initialize counter ---
SET COUNTER=0
SET TOTAL=0

REM --- Count the total number of directories ---
FOR /D %%D in ("%FOLDER_PATH%\%FILENAME_PREFIX%*") DO (
    SET /A TOTAL+=1
)

REM --- Loop through directories starting with the prefix and run in parallel ---
FOR /D %%D in ("%FOLDER_PATH%\%FILENAME_PREFIX%*") DO (
    SET /A COUNTER+=1
    FOR %%I IN (%%D) DO SET LAST_SUBDIR=%%~nxI
    ECHO Starting agent !COUNTER!/!TOTAL! at directory: !LAST_SUBDIR!
    start "Agent: !LAST_SUBDIR!" cmd /k python "%PYTHON_SCRIPT%" --prefix "!LAST_SUBDIR!"
    ECHO Started process for agent: !COUNTER!/!TOTAL!
    ECHO ===================================================================
)

ECHO All processes started. Check individual process outputs for completion.
pause