from setuptools import setup, find_packages

setup(
    name="driver-awareness",
    version="0.1.0",
    # Tells setuptools to find packages in the 'src' directory
    packages=find_packages(where="src") + ["."], 
    package_dir={"": "src", "camera": "."},
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
    ],
    python_requires=">=3.8",
    description="A modular real-time driver behavior analysis system.",
    author="Armaan",
)
