"""Launch a research script from the repository root with local imports resolved."""
import os,runpy,sys
from pathlib import Path
from wearflow_paths import ROOT

def main():
    if len(sys.argv)<2 or sys.argv[1] in ('-h','--help'):
        print('Usage: python -m wearflow.run experiments/<dataset>/<script.py> [script arguments]')
        return
    script=(ROOT/sys.argv[1]).resolve()
    if not script.is_relative_to(ROOT) or not script.is_file():raise SystemExit('Expected a script inside this checkout')
    os.chdir(ROOT);sys.path.insert(0,str(script.parent));sys.argv=[str(script),*sys.argv[2:]]
    runpy.run_path(str(script),run_name='__main__')
if __name__=='__main__':main()
