@echo off

for %%f in ("data\ucf\Fighting\*.mp4") do (
    echo Processing: %%~nf
    python run.py --video "%%f" ^
                  --zones "config\zones_template.json" ^
                  --output "results\ucf_fighting\%%~nf\"
)

echo Done processing all UCF videos.
pause