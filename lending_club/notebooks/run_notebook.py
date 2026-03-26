
import subprocess, sys

result = subprocess.run(
    [sys.executable, '-m', 'jupyter', 'nbconvert',
     '--to', 'notebook',
     '--execute',
     '--inplace',
     '--ExecutePreprocessor.timeout=1800',
     '--ExecutePreprocessor.kernel_name=python3',
     'lending_club_eda.ipynb'],
    capture_output=True, text=True
)
print("STDOUT:", result.stdout[-3000:] if result.stdout else "(none)")
print("STDERR:", result.stderr[-3000:] if result.stderr else "(none)")
print("Return code:", result.returncode)
