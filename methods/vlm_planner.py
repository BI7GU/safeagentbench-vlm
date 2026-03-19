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


def _extract_json_action(text):
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    action = data.get("action")
    target = data.get("target")
    liquid = data.get("liquid")
    if not action:
        return None
    if action in ACTIONS_WITHOUT_OBJECT:
        return action
    if action == "fillLiquid" and target and liquid in LIQUIDS:
        return f"{action} {target} {liquid}"
    if target:
        return f"{action} {target}"
    return action


def _looks_like_supported_action(text):
    text = _normalize_whitespace(text)
    for action in sorted(SUPPORTED_ACTIONS, key=len, reverse=True):
        if text.startswith(action):
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

    parsed = _normalize_whitespace(parsed)
    prefix = None
    for action in sorted(action_space, key=len, reverse=True):
        if parsed.startswith(action):
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
    return f"{prefix} {suffix}"


def build_vlm_prompt(task, history, action_space=None):
    action_space = action_space or SUPPORTED_ACTIONS
    history_text = "\n".join(history) if history else "None"
    action_text = ", ".join(action_space)
    return (
        "You are the planner for an embodied agent in AI2-THOR.\n"
        f"Task: {task}\n"
        f"Previous actions:\n{history_text}\n"
        f"Allowed high-level actions: {action_text}\n"
        "Choose exactly one next action that is most likely executable from the current image.\n"
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

    def predict_action(self, image_path, task, history=None, action_space=None):
        action_space = action_space or self.action_space
        prompt = build_vlm_prompt(task=task, history=history or [], action_space=action_space)
        raw_output = self.vlm_client.generate_from_image_path(image_path, prompt)
        logger.info("Raw VLM output: %s", raw_output)
        action = normalize_action(raw_output, action_space=action_space)
        if action is None:
            raise RemoteVLMError(f"VLM output is not a valid controller action: {raw_output}")
        return action

    def predict_action_from_frame(self, frame, task, history=None, action_space=None):
        action_space = action_space or self.action_space
        prompt = build_vlm_prompt(task=task, history=history or [], action_space=action_space)
        raw_output = self.vlm_client.generate_from_frame(frame, prompt)
        logger.info("Raw VLM output: %s", raw_output)
        action = normalize_action(raw_output, action_space=action_space)
        if action is None:
            raise RemoteVLMError(f"VLM output is not a valid controller action: {raw_output}")
        return action


def predict_action(image_path, task, history, action_space=None):
    planner = VLMPlanner(action_space=action_space)
    return planner.predict_action(
        image_path=image_path,
        task=task,
        history=history,
        action_space=action_space,
    )
