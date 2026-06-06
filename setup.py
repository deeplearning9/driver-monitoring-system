from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """读取 requirements.txt 文件"""
    requirements = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # 移除版本注释
                if '#' in line:
                    line = line[:line.index('#')].strip()
                requirements.append(line)
    return requirements

setup(
    name="driver-monitoring-system",
    version="1.0.0",
    author="deeplearning9",
    author_email="1724022286@qq.com",
    description="基于深度学习的汽车座舱智能监控系统 - 驾驶员状态监控与手势识别",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deeplearning9/driver-monitoring-system",
    project_urls={
        "Bug Tracker": "https://github.com/deeplearning9/driver-monitoring-system/issues",
        "Documentation": "https://github.com/deeplearning9/driver-monitoring-system/tree/main/docs",
        "Source Code": "https://github.com/deeplearning9/driver-monitoring-system",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.23.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "tensorrt>=8.6.0",  # 需要 NVIDIA GPU
        ],
    },
    entry_points={
        "console_scripts": [
            "dms-train=training.train_detection:main",
            "dms-infer=inference.realtime_monitor:main",
            "dms-web=webapp.app:main",
            "dms-demo=inference.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "deep-learning",
        "computer-vision",
        "driver-monitoring",
        "gesture-recognition",
        "fatigue-detection",
        "yolo",
        "pytorch",
        "opencv",
    ],
)
