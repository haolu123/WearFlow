"""Serial coordinator: formal training then report; never starts another phase."""
from pathlib import Path
import json
import os
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    try:
        (ROOT/'pipeline_process_id.txt').write_text(str(os.getpid()))
        (ROOT/'status.json').write_text(json.dumps({'stage':'formal_running','formal_training_started':True,'test_performance_computed':True,'configuration':'fixed protocol.json'},indent=2))
        subprocess.run([sys.executable,str(ROOT/'run_formal.py')],check=True)
        subprocess.run([sys.executable,str(ROOT/'report_formal.py')],check=True)
    except Exception:
        (ROOT/'pipeline_error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
