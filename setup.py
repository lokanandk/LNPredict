from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lnp-hyperparameter-tuning",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Comprehensive framework for LNP property prediction with hyperparameter tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/lnp-hyperparameter-tuning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lnp-train=scripts.train:main",
            "lnp-evaluate=scripts.evaluate:main",
            "lnp-hypersearch=scripts.hyperparameter_search:main",
            "lnp-predict=scripts.predict:main",
        ],
    },
)
