import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='OscillAI',
    version='0.0.1',
    author='Giorgio Morales - GREYC',
    author_email='giorgiomorales@ieee.org',
    description='Simulating neutrino oscillation maps using AI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/GiorgioMorales/OscillationMaps',
    project_urls={"Bug Tracker": "https://github.com/GiorgioMorales/OscillationMaps/issues"},
    license='MIT',
    packages=setuptools.find_packages('src', exclude=['test']),
    # packages=setuptools.find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=['matplotlib==3.9', 'numpy', 'opencv-python', 'tqdm', 'h5py', 'pyodbc', 'regex',
                      'torchsummary', 'python-dotenv', 'omegaconf', 'pandas', 'pynvml'],
)
