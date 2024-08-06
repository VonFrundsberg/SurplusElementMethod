import os

def create_init_files(root_dir):
    # Create or update the root __init__.py
    root_init_path = os.path.join(root_dir, '__init__.py')
    with open(root_init_path, 'w') as root_init_file:
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path) and not subdir.startswith('__'):
                root_init_file.write(f'from .{subdir} import *\n')

    # Recursively create __init__.py in subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__init__.py' not in filenames:
            with open(os.path.join(dirpath, '__init__.py'), 'w') as init_file:
                for filename in filenames:
                    if filename.endswith('.py') and filename != '__init__.py':
                        module_name = filename[:-3]
                        init_file.write(f'from .{module_name} import *\n')

# Run the script
create_init_files('SurplusElement')