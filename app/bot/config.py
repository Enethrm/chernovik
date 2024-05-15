from configparser import ConfigParser
import os

cfg = ConfigParser() 
cfg.read(os.path.join(os.path.dirname(__file__), 'config.ini'),encoding='utf-8')

