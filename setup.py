from setuptools import setup

install_deps = ['cellpose', 'pandas', 'numpy', 'scipy', 'scikit-image', 'pillow']

setup(name='cellstats',
      version='0.1',
      description='A tool for analyzing microscopy images',
      url='https://github.com/noamblum/cellstats.git',
      author='Noam Blum',
      author_email='noam.blum1@mail.huji.ac.il',
      license='BSD',
      packages=['cellstats'],
      install_requires=install_deps,
      zip_safe=False,
      entry_points = {
        'console_scripts': [
            'cellstats = cellstats.__main__:main'
        ]
      })
