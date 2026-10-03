@echo off
rem Bam dup de chay tracking tay gang VA ghi lai anh (--record) vao recordings\<gio> de phan tich.
rem Ban ghi nang ~200 MB/phut -- chi dung khi can.
cd /d "%~dp0"
"%~dp0mmpose_venv\Scripts\python.exe" run_glove_quest_stream.py --checkpoint checkpoints\rtmpose_glove_pinch_color3.pth --record %*
echo.
echo Da dung. Nhan phim bat ky de dong cua so.
pause >nul
