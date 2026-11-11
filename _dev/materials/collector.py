"""Optional scripted correction applied before each recorded environment action."""


def correct_action(observation, proposed_action, step):
    """Return a finite 7D OSC_POSE action; the default keeps the policy output.

    Edit this file to test an explicit scripted correction. These records are
    policy/script generated, not human teleoperation demonstrations.
    """
    return proposed_action
