import numpy as np
import os
import yaml
import pandas as pd

def normalize(img):
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
    return img

def read_text(filename):
    df = pd.read_excel(filename)

    # row is a dictionary: {'Image': ..., 'Ground Truth': ..., 'Description': ...}
    return df.to_dict(orient="records")

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())
    
    def merge_from_list(self, opts):
        """
        Merge config options from a list like:
        ["MODEL.CLIP_MODEL", "unimedclip", "TRAIN.BATCH_SIZE", "16"]
        """
        if opts is None:
            return

        if len(opts) % 2 != 0:
            raise ValueError("opts must be key-value pairs")

        for full_key, v in zip(opts[0::2], opts[1::2]):
            key_list = full_key.split(".")

            cur = self
            for k in key_list[:-1]:
                if k not in cur:
                    raise KeyError(f"Invalid config key: {full_key}")
                cur = cur[k]

            final_key = key_list[-1]
            if final_key not in cur:
                raise KeyError(f"Invalid config key: {full_key}")

            # -------- type inference --------
            if isinstance(v, str):
                vl = v.lower()
                if vl == "true":
                    v = True
                elif vl == "false":
                    v = False
                else:
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass

            cur[final_key] = v


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:

        # for k, v in cfg_from_file[key].items():
        cfg[key] = cfg_from_file[key]

    cfg = CfgNode(cfg)

    return cfg