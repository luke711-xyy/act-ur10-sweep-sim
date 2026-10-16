"""Explicit PAC-ACT lifecycle; importing or running without flags is read-only."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pac_act.cli import main

if __name__=='__main__': raise SystemExit(main())
