from setuptools import setup, find_packages

setup(name='fastsmt',
      version='1.0',
      description='FastSMT',
      url='http://fastsmt.ethz.ch/',
      author='ETH SRI',
      author_email='bmislav@ethz.ch',
      license='MIT',
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'torch',
          'matplotlib',
          'tensorboardX',
          'seaborn',
      ],
      packages=find_packages(),
)
