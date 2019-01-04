import os.path
import re

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
    version_content = [line for line in f.readlines() if re.search(r'([\d.]+)',line)]

if len(version_content) != 1:
    raise RuntimeError('Invalid format of VERSION file.')

__version__ = version_content[0]
