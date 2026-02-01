import sys
import os
# Ensure the current directory is in sys.path so we can import qwen_asr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_asr.cli.demo import main

# Paths
MODEL_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ASR-1.7B"
ALIGNER_DIR = r"C:\Users\Haujet\.cache\modelscope\hub\models\Qwen\Qwen3-ForcedAligner-0.6B"

def run():
    # Fixed parameters
    args = [
        "--asr-checkpoint", MODEL_DIR,
        "--ip", "127.0.0.1",
        "--port", "8000",
        "--concurrency", "4"
    ]
    
    # Check if aligner exists and add it if so
    if os.path.exists(ALIGNER_DIR):
        print(f"Aligner found at {ALIGNER_DIR}, enabling timestamps.")
        args.extend(["--aligner-checkpoint", ALIGNER_DIR])
    else:
        print(f"Aligner not found at {ALIGNER_DIR}, running without alignment.")

    print(f"Starting Qwen3-ASR Demo...")
    # Inject arguments into sys.argv
    sys.argv = ["qwen-asr-demo"] + args
    
    main()

if __name__ == "__main__":
    run()
