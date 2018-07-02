import os
import importlib.machinery

def load_modules(module_names, project_name='muellericcv2017'):

    modules = {}
    for module_name in module_names:
        abs_path_module = os.path.abspath(os.path.join(__file__, '..', '..')) +\
                          '/' + project_name + '/' + module_name + '.py'
        loader = importlib.machinery.SourceFileLoader(module_name, abs_path_module)
        module = loader.load_module(module_name)
        modules[module_name] = module
    return modules
