import json
import logging
import re

from vlm import RemoteVLMClient, RemoteVLMError


logger = logging.getLogger(__name__)

SUPPORTED_ACTIONS = [
    "find",
    "pick",
    "put",
    "open",
    "close",
    "slice",
    "turn on",
    "turn off",
    "drop",
    "throw",
    "break",
    "cook",
    "dirty",
    "clean",
    "fillLiquid",
    "emptyLiquid",
    "pour",
]

ACTIONS_WITHOUT_OBJECT = {"drop", "throw", "pour"}
LIQUIDS = {"water", "coffee", "wine"}


def _normalize_whitespace(text):
    return " ".join((text or "").strip().split())


def _clean_action_text(text):
    text = _normalize_whitespace(text)
    replacements = {
        "turn_on ": "turn on ",
        "turn_off ": "turn off ",
        "toggle_on ": "turn on ",
        "toggle_off ": "turn off ",
        "fill liquid ": "fillLiquid ",
        "fillliquid ": "fillLiquid ",
        "empty liquid ": "emptyLiquid ",
        "emptyliquid ": "emptyLiquid ",
        "pickup ": "pick ",
        "pick up ": "pick ",
        "pickup the ": "pick ",
        "turn the ": "turn ",
    }
    lowered = text.lower()
    for src, dst in replacements.items():
        if lowered.startswith(src):
            text = dst + text[len(src):]
            break
    return text.strip(" .,\n\t")


def _rewrite_natural_action(text):
    text = _clean_action_text(text)
    lowered = text.lower()

    if lowered.startswith(("place ", "put ")) and any(token in lowered for token in [" into ", " in ", " inside ", " on ", " onto "]):
        for token in [" into ", " inside ", " onto ", " on ", " in "]:
            if token in lowered:
                idx = lowered.index(token)
                target = text[idx + len(token):].strip(" .,\n\t")
                if target:
                    return f"put {target}"

    if lowered.startswith(("drop ", "throw ", "pour ")):
        verb = lowered.split()[0]
        return verb

    if lowered.startswith("fill "):
        parts = text.split()
        if len(parts) >= 3 and parts[-1].lower() in LIQUIDS:
            target = " ".join(parts[1:-1]).strip(" .,\n\t")
            liquid = parts[-1].lower()
            if target:
                return f"fillLiquid {target} {liquid}"

    if lowered.startswith("fillliquid ") or lowered.startswith("fillLiquid ".lower()):
        parts = text.split()
        if len(parts) >= 3 and parts[-1].lower() in LIQUIDS:
            target = " ".join(parts[1:-1]).strip(" .,\n\t")
            liquid = parts[-1].lower()
            if target:
                return f"fillLiquid {target} {liquid}"

    if lowered.startswith("empty "):
        target = text[6:].strip(" .,\n\t")
        if target:
            return f"emptyLiquid {target}"

    if lowered.startswith(("close ", "open ", "slice ", "pick ", "find ", "break ", "cook ", "dirty ", "clean ")):
        return text

    if lowered.startswith("turn on "):
        return "turn on " + text[8:].strip(" .,\n\t")

    if lowered.startswith("turn off "):
        return "turn off " + text[9:].strip(" .,\n\t")

    if lowered.startswith("toggle on "):
        return "turn on " + text[10:].strip(" .,\n\t")

    if lowered.startswith("toggle off "):
        return "turn off " + text[11:].strip(" .,\n\t")

    return text


def _extract_json_action(text):
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    action = data.get("action")
    target = data.get("target") or data.get("object") or data.get("receptacle")
    liquid = data.get("liquid")
    if not action:
        return None
    action = _rewrite_natural_action(str(action))
    if action.lower() == "fillliquid":
        action = "fillLiquid"
    if action.lower() == "emptyliquid":
        action = "emptyLiquid"
    if action in ACTIONS_WITHOUT_OBJECT:
        return action
    if action == "fillLiquid" and target and str(liquid).lower() in LIQUIDS:
        return f"{action} {target} {str(liquid).lower()}"
    if target:
        return f"{action} {target}"
    return action


def _looks_like_supported_action(text):
    text = _rewrite_natural_action(text)
    for action in sorted(SUPPORTED_ACTIONS, key=len, reverse=True):
        if text.lower().startswith(action.lower()):
            return text
    return None


def normalize_action(raw_text, action_space=None):
    action_space = action_space or SUPPORTED_ACTIONS
    parsed = _extract_json_action(raw_text)
    if parsed is None:
        parsed = _looks_like_supported_action(raw_text)
    if parsed is None:
        logger.warning("Could not parse VLM output into a supported action: %s", raw_text)
        return None

    parsed = _rewrite_natural_action(parsed)
    prefix = None
    for action in sorted(action_space, key=len, reverse=True):
        if parsed.lower().startswith(action.lower()):
            prefix = action
            break
    if prefix is None:
        logger.warning("Parsed action is outside whitelist: %s", parsed)
        return None

    suffix = parsed[len(prefix):].strip()
    if prefix in ACTIONS_WITHOUT_OBJECT:
        return prefix
    if prefix == "fillLiquid":
        parts = suffix.split()
        if len(parts) < 2 or parts[-1] not in LIQUIDS:
            logger.warning("Invalid fillLiquid action: %s", parsed)
            return None
        return f"{prefix} {' '.join(parts[:-1])} {parts[-1]}"
    if not suffix:
        logger.warning("Action requires a target object: %s", parsed)
        return None
    return f"{prefix} {suffix.strip(' .')}"


def build_vlm_prompt(task, history, action_space=None, reference_steps=None):
    action_space = action_space or SUPPORTED_ACTIONS
    history_text = "\n".join(history) if history else "None"
    action_text = ", ".join(action_space)
    reference_text = "\n".join(reference_steps) if reference_steps else "None"
    return (
        "You are the planner for an embodied agent in AI2-THOR.\n"
        f"Task: {task}\n"
        f"Previous actions:\n{history_text}\n"
        f"Reference actions for this task family:\n{reference_text}\n"
        f"Allowed high-level actions: {action_text}\n"
        "Choose exactly one next action that is most likely executable from the current image.\n"
        "Prefer actions that stay consistent with the task and the reference action style.\n"
        "Output either:\n"
        '1. A single action string, like "find Cabinet" or "turn on DeskLamp"\n'
        '2. JSON like {"action": "find", "target": "Cabinet"}\n'
        'For drop/throw/pour, output only the action word.\n'
        "Do not explain your reasoning. Do not output multiple steps."
    )


class VLMPlanner:
    def __init__(self, vlm_client=None, action_space=None):
        self.vlm_client = vlm_client or RemoteVLMClient()
        self.action_space = action_space or SUPPORTED_ACTIONS

    def predict_action(self, image_path, task, history=None, action_space=None, reference_steps=None):
        action_space = action_space or self.action_space
        prompt = build_vlm_prompt(
            task=task,
            history=history or [],
            action_space=action_space,
            reference_steps=reference_steps,
        )
        info = self.predict_action_with_info(
            image_path=image_path,
            task=task,
            history=history,
            action_space=action_space,
            reference_steps=reference_steps,
        )
        if info["action"] is None:
            raise RemoteVLMError(f"VLM output is not a valid controller action: {info['raw_output']}")
        return info["action"]

    def predict_action_from_frame(self, frame, task, history=None, action_space=None, reference_steps=None):
        info = self.predict_action_from_frame_with_info(
            frame=frame,
            task=task,
            history=history,
            action_space=action_space,
            reference_steps=reference_steps,
        )
        if info["action"] is None:
            raise RemoteVLMError(f"VLM output is not a valid controller action: {info['raw_output']}")
        return info["action"]

    def predict_action_with_info(self, image_path, task, history=None, action_space=None, reference_steps=None):
        action_space = action_space or self.action_space
        prompt = build_vlm_prompt(
            task=task,
            history=history or [],
            action_space=action_space,
            reference_steps=reference_steps,
        )
        raw_output = self.vlm_client.generate_from_image_path(image_path, prompt)
        logger.info("Raw VLM output: %s", raw_output)
        action = normalize_action(raw_output, action_space=action_space)
        return {
            "raw_output": raw_output,
            "normalized_output": _rewrite_natural_action(raw_output),
            "action": action,
            "parse_error": None if action is not None else "unsupported_or_unparseable_output",
        }

    def predict_action_from_frame_with_info(self, frame, task, history=None, action_space=None, reference_steps=None):
        action_space = action_space or self.action_space
        prompt = build_vlm_prompt(
            task=task,
            history=history or [],
            action_space=action_space,
            reference_steps=reference_steps,
        )
        raw_output = self.vlm_client.generate_from_frame(frame, prompt)
        logger.info("Raw VLM output: %s", raw_output)
        action = normalize_action(raw_output, action_space=action_space)
        return {
            "raw_output": raw_output,
            "normalized_output": _rewrite_natural_action(raw_output),
            "action": action,
            "parse_error": None if action is not None else "unsupported_or_unparseable_output",
        }


def predict_action(image_path, task, history, action_space=None, reference_steps=None):
    planner = VLMPlanner(action_space=action_space)
    return planner.predict_action(
        image_path=image_path,
        task=task,
        history=history,
        action_space=action_space,
        reference_steps=reference_steps,
    )
