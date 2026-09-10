@echo off
rem 启动 GLP-1 看板（自动定位脚本所在目录，可移植）
cd /d "%~dp0"
start "GLP1-Dashboard" cmd /k "python -m streamlit run dashboard/app.py --server.headless true --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false"
