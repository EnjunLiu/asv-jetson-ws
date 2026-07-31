import sys, os

PROJECT = r"C:\Users\LIU\Documents\jetson_ws\asv_vla"
sys.path.insert(0, os.path.join(PROJECT, "src", "asv_vla"))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from training.evaluate_selection import main
sys.exit(main())
