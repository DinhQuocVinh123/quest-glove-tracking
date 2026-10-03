@echo off
rem Bam dup de chay tracking tay gang (khong ghi anh).
rem Tu vao dung thu muc va dung dung Python cua du an (mmpose_venv) -- khong can go lenh.
rem Them tuy chon thi keo file nay vao cua so cmd roi go them, vd: --flip
cd /d "%~dp0"
"%~dp0mmpose_venv\Scripts\python.exe" run_glove_quest_stream.py --checkpoint checkpoints\rtmpose_glove_pinch_color3.pth %*
echo.
echo Da dung. Nhan phim bat ky de dong cua so.
pause >nul
