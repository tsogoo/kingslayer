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