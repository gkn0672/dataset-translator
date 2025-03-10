import os
import subprocess
import shutil


def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # List all only directories in the script directory
    dirs = [
        d for d in os.listdir(script_dir) if os.path.isdir(os.path.join(script_dir, d))
    ]
    non_build_dirs = ["__pycache__", "build", "dist", "string_ops.egg-info"]
    dirs = [d for d in dirs if d not in non_build_dirs]
    for dir in dirs:
        print(f"Building {dir}")
        # Set the target directory
        target_dir = os.path.join(script_dir, dir)

        # Ensure we're in the correct directory
        os.chdir(target_dir)

        # Run the build command
        subprocess.check_call(["python", "setup.py", "build_ext", "--inplace"])

        # Find and move the .so files if they're not already in place
        build_dir = os.path.join(target_dir, "build")
        if os.path.exists(build_dir):
            for root, dirs, files in os.walk(build_dir):
                for file in files:
                    if file.endswith(".so"):
                        src = os.path.join(root, file)
                        dst = os.path.join(target_dir, file)
                        shutil.move(src, dst)
                        print(f"Moved {src} to {dst}")

            # Clean up build directory
            shutil.rmtree(build_dir)
            print(f"Removed build directory: {build_dir}")
        else:
            print("No build directory found. .so files should already be in place.")


if __name__ == "__main__":
    main()
