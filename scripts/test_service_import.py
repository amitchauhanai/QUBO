import importlib
m = importlib.import_module('qubo.service')
print('status():', m.status())
