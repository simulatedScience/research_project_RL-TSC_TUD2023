@echo off
for /L %%i in (1,1,12) do ( @REM run 12 times
    python run_train.py
)