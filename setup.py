import setuptools

setuptools.setup(
    name="asym",
    version="0.0.1",
    author="The PD-1 team",
    author_email="clemens_hug@hms.harvard.edu",
    description="Detect assymetry in cell markers",
    url="https://github.com/IAWG-CSBC-PSON/pd1-asymmetry",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "click",
        "numpy",
        "pandas",
        "pillow",
        "bokeh",
        "torch",
        "matplotlib",
        "umap",
    ],
    scripts=["bin/asym"],
)
