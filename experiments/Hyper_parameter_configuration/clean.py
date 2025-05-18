output_dir = "/workspace/outputs"
unsloth_cache = "/workspace/unsloth_compiled_cache"
import shutil



for dir_path in [output_dir, unsloth_cache]:
        print(f"\nAttempting to delete {dir_path}...")
        shutil.rmtree(dir_path, ignore_errors=False)
        print(f"Successfully deleted {dir_path}")
            