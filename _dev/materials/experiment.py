"""One participant-editable experiment. Evaluation conditions stay fixed."""


def configure(config):
    """Example: execute two actions per model chunk instead of one."""
    config["policy"]["n_action_steps"] = 2
    # To evaluate a learned checkpoint instead:
    # config["policy"]["path"] = "/content/drive/MyDrive/.../pretrained_model"
    # config["policy"]["revision"] = None
    return config
