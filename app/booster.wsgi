import logging
import sys, os
from os.path import split, join, dirname
import site
sys.stdout = sys.stderr

debug = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filename='/tmp/booster.log', filemode='w')
logging.info(' > Booster Web App starting ...')

# - Find path to site-packages directory
python_home = dirname(split(sys.executable)[0])
python_version = '.'.join(map(str, sys.version_info[:2]))
site_packages = join(python_home, f'lib/python{python_version}/site-packages')

if debug:
    print(f' >>> {python_version} ::: {site_packages}')

site.addsitedir(site_packages)
prev_sys_path = list(sys.path)
site.addsitedir(site_packages)
new_sys_path = []

for item in list(sys.path):
    if item not in prev_sys_path:
        new_sys_path.append(item)
        sys.path.remove(item)

sys.path[:0] = new_sys_path

if debug:
    print(f' >>> start application loading, cwd : {os.getcwd()} ')

from qube.booster.app.boo import app as application
if debug:
    print(f' >>> DONE')