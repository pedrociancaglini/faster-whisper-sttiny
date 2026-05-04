@echo off
setlocal

:: Define the folder containing .mp4 files
set "input_folder=C:\Users\Public\Videos"
set "faster_whisper_path=C:\Users\Public\Faster-Whisper-XXL\faster-whisper-xxl.exe"
set "output_dir=C:\Users\Public\Faster-Whisper-Output"

:: Change to the input folder
cd /d "%input_folder%"

:: Loop through all .mp4 files in the folder
for %%f in (*.mp4) do (
    echo Processing: %%f
    "%faster_whisper_path%" "%%f" --language Spanish --model large-v2 --output_dir "%output_dir%" --output_format text --without_timestamps True --word_timestamps False --device cpu --threads 8
)

echo All files processed.
pause
