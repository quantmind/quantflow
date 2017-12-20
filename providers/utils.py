import os
import json
import configparser


def user_agent_info():
    return ''


def from_config(keys, entry=None, config_file=None):
    config = configparser.ConfigParser()
    config_file = config_file or os.environ.get('CONFIG_FILE')
    assert config_file, "CONFIG_FILE is not available"
    path = os.path.join(os.path.expanduser("~"), config_file)
    if not os.path.isfile(path):
        return keys
    if config_file.endswith('.json'):
        with open(config_file) as fp:
            data = json.load(fp)
        config = data.get(entry or 'default') or {}
        for key, value in keys.items():
            if not value:
                keys[key] = config.get(key)
    else:
        config.read(path)
        entry = entry or 'default'
        for key, value in keys.items():
            if not value:
                keys[key] = config.get(entry, key)
    return keys
