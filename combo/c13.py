"""C13 organizer-data variant: C10 x0.85 + h6 x0.075 + h10 x0.075, CSLS then Sinkhorn.
The two networks have six and ten targets; neither uses text descriptions.
"""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('NC_DIR', os.path.join(HERE, 'c13_inputs'))
os.environ.setdefault('NC_FAMILIES', 'h10:0.075,h6:0.075')
sys.argv[0] = os.path.join(HERE, 'c11.py')
runpy.run_path(os.path.join(HERE, 'c11.py'), run_name='__main__')
