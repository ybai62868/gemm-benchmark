import importlib.machinery

def load_function_from_module(module_name, function_name):
    loader = importlib.machinery.SourceFileLoader(module_name, module_name + ".py")
    module = loader.load_module()
    if hasattr(module, function_name):
        return getattr(module, function_name)
    else:
        raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'")
