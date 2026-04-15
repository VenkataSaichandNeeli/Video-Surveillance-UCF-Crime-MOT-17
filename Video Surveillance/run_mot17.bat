@echo off

for /d %%d in ("data\MOT17\*") do (
    echo Processing sequence: %%~nxd
    python run.py --video "%%d" ^
                  --zones "config\zones_template.json" ^
                  --output "results\mot17\%%~nxd\" ^
                  --mode sequence ^
                  --evaluate
)

echo Done processing all MOT17 sequences.
pause