def get_config(conf:dict, config:str):
    confs = config.split(":")
    v = conf
    if len(confs) > 0:
        for conf in confs:
            if conf in v:
                v = v[conf]
            else:
                return None
    return v

def load_config():
    from pathlib import Path
    current_file_path = Path(__file__).resolve()
    root_directory = current_file_path.parents[1]
    file_path = root_directory / 'app.yaml'
    import yaml
    with open(file_path, "r") as file:
        return yaml.safe_load(file)