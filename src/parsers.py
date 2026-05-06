import csv
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src._yaml import get_yaml
from src.schemas import (
    BlueprintPhaseCreate,
    ExperimentBlueprintCreate,
    ParticipantProfileCreate,
)

yaml = get_yaml()

logger = logging.getLogger(__name__)


class QuestionnaireParsingError(Exception):
    """Raised when the questionnaire file cannot be parsed."""

    pass


class BlueprintParsingError(Exception):
    """Raised when the blueprint file cannot be parsed."""

    pass


class BaseQuestionnaireParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> ParticipantProfileCreate:
        pass


class BaseBlueprintParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> ExperimentBlueprintCreate:
        pass


def _parse_list_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    parts = [segment.strip() for segment in re.split(r"\||;|,", text)]
    return [segment for segment in parts if segment]


def _parse_preferences_field(value: Any) -> Dict[str, float]:
    if value is None:
        return {}

    if isinstance(value, dict):
        result: Dict[str, float] = {}
        for key, raw_val in value.items():
            try:
                result[str(key)] = float(raw_val)
            except (ValueError, TypeError):
                continue
        return result

    text = str(value).strip()
    if not text:
        return {}

    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, dict):
            return _parse_preferences_field(parsed_json)
    except json.JSONDecodeError:
        pass

    prefs: Dict[str, float] = {}
    for chunk in re.split(r"\||;|,", text):
        pair = chunk.strip()
        if not pair or ":" not in pair:
            continue
        key, raw_val = pair.split(":", 1)
        key = key.strip()
        if not key:
            continue
        try:
            prefs[key] = float(raw_val.strip())
        except ValueError:
            continue
    return prefs


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_modalities_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    parts = [segment.strip() for segment in re.split(r"\||;|,", text)]
    return [segment for segment in parts if segment]


def _to_profile_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if "participant_id" in data and "id" not in data:
        data["id"] = data.get("participant_id")

    profile_data = {
        "id": data.get("id"),
        "baseline_valence": float(data.get("baseline_valence", 0.0)),
        "baseline_arousal": float(data.get("baseline_arousal", 0.0)),
        "phobias": _parse_list_field(data.get("phobias", [])),
        "familiar_media": _parse_list_field(data.get("familiar_media", [])),
        "preferences": _parse_preferences_field(data.get("preferences", {})),
        "is_excluded": _as_bool(data.get("is_excluded", False)),
        "exclusion_reason": data.get("exclusion_reason", None),
    }

    if not profile_data["id"]:
        raise QuestionnaireParsingError("Missing 'id' or 'participant_id'")

    return profile_data


def _to_blueprint_data(data: Dict[str, Any]) -> Dict[str, Any]:
    blueprint_id = data.get("id") or data.get("experiment_id")
    blueprint_name = data.get("name") or blueprint_id

    if not blueprint_id:
        raise BlueprintParsingError("Missing blueprint 'id' or 'experiment_id'")

    phases_raw = data.get("phases")
    if not isinstance(phases_raw, list) or not phases_raw:
        raise BlueprintParsingError("Blueprint must include at least one phase")

    phases: List[Dict[str, Any]] = []
    for idx, phase in enumerate(phases_raw, start=1):
        if not isinstance(phase, dict):
            raise BlueprintParsingError(f"Phase at index {idx} must be an object")

        phase_order = phase.get("phase_order", idx)
        phase_data = {
            "phase_order": int(phase_order),
            "target_valence": float(phase.get("target_valence", 0.0)),
            "target_arousal": float(phase.get("target_arousal", 0.0)),
            "min_duration_ms": int(phase.get("min_duration_ms", 1000)),
            "max_duration_ms": int(phase.get("max_duration_ms", 1000)),
            "allowed_modalities": _parse_modalities_field(phase.get("allowed_modalities", [])),
        }
        if not phase_data["allowed_modalities"]:
            raise BlueprintParsingError(f"Phase {phase_order} must include allowed_modalities")
        phases.append(phase_data)

    return {
        "id": str(blueprint_id),
        "name": str(blueprint_name),
        "description": data.get("description"),
        "phases": phases,
    }


class JSONQuestionnaireParser(BaseQuestionnaireParser):
    def parse(self, content: str) -> ParticipantProfileCreate:
        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                raise QuestionnaireParsingError("JSON questionnaire must be an object")
            profile_data = _to_profile_data(data)
            return ParticipantProfileCreate(**profile_data)
        except json.JSONDecodeError as e:
            raise QuestionnaireParsingError(f"Invalid JSON: {str(e)}")
        except Exception as e:
            raise QuestionnaireParsingError(f"Error parsing JSON questionnaire: {str(e)}")


class XMLQuestionnaireParser(BaseQuestionnaireParser):
    def parse(self, content: str) -> ParticipantProfileCreate:
        try:
            root = ET.fromstring(content)

            p_id = root.findtext("participant_id") or root.findtext("id")
            if not p_id:
                raise QuestionnaireParsingError("Missing <participant_id> or <id> in XML")

            familiar_media = []
            familiar_node = root.find("familiar_media")
            if familiar_node is not None:
                familiar_media = [item.text for item in familiar_node.findall("item") if item.text]

            # Handle other optional fields
            baseline_val = float(root.findtext("baseline_valence", "0.0"))
            baseline_aro = float(root.findtext("baseline_arousal", "0.0"))

            phobias = []
            phobias_node = root.find("phobias")
            if phobias_node is not None:
                phobias = [p.text for p in phobias_node.findall("phobia") if p.text]

            preferences = {}
            prefs_node = root.find("preferences")
            if prefs_node is not None:
                for pref in prefs_node.findall("preference"):
                    stim_id = pref.get("id")
                    if stim_id:
                        try:
                            if pref.text is None:
                                continue
                            preferences[stim_id] = float(pref.text)
                        except (ValueError, TypeError):
                            continue

            return ParticipantProfileCreate(
                id=p_id,
                baseline_valence=baseline_val,
                baseline_arousal=baseline_aro,
                phobias=phobias,
                familiar_media=familiar_media,
                preferences=preferences,
            )
        except ET.ParseError as e:
            raise QuestionnaireParsingError(f"Invalid XML: {str(e)}")
        except Exception as e:
            raise QuestionnaireParsingError(f"Error parsing XML questionnaire: {str(e)}")


class CSVQuestionnaireParser(BaseQuestionnaireParser):
    def parse(self, content: str) -> ParticipantProfileCreate:
        try:
            stream = io.StringIO(content)
            reader = csv.DictReader(stream)
            rows = list(reader)

            if not rows:
                raise QuestionnaireParsingError(
                    "CSV questionnaire must include a header and at least one row"
                )

            profile_data = _to_profile_data(rows[0])
            return ParticipantProfileCreate(**profile_data)
        except QuestionnaireParsingError:
            raise
        except Exception as e:
            raise QuestionnaireParsingError(f"Error parsing CSV questionnaire: {str(e)}")


class YAMLQuestionnaireParser(BaseQuestionnaireParser):
    def parse(self, content: str) -> ParticipantProfileCreate:
        if yaml is None:
            raise QuestionnaireParsingError(
                "YAML support is unavailable because PyYAML is not installed"
            )
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                raise QuestionnaireParsingError("YAML questionnaire must be a mapping")
            profile_data = _to_profile_data(data)
            return ParticipantProfileCreate(**profile_data)
        except QuestionnaireParsingError:
            raise
        except Exception as e:
            raise QuestionnaireParsingError(f"Error parsing YAML questionnaire: {str(e)}")


class JSONBlueprintParser(BaseBlueprintParser):
    def parse(self, content: str) -> ExperimentBlueprintCreate:
        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                raise BlueprintParsingError("JSON blueprint must be an object")
            blueprint_data = _to_blueprint_data(data)
            return ExperimentBlueprintCreate(
                **{
                    **blueprint_data,
                    "phases": [BlueprintPhaseCreate(**phase) for phase in blueprint_data["phases"]],
                }
            )
        except json.JSONDecodeError as e:
            raise BlueprintParsingError(f"Invalid JSON: {str(e)}")
        except BlueprintParsingError:
            raise
        except Exception as e:
            raise BlueprintParsingError(f"Error parsing JSON blueprint: {str(e)}")


class YAMLBlueprintParser(BaseBlueprintParser):
    def parse(self, content: str) -> ExperimentBlueprintCreate:
        if yaml is None:
            raise BlueprintParsingError(
                "YAML support is unavailable because PyYAML is not installed"
            )
        try:
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                raise BlueprintParsingError("YAML blueprint must be a mapping")
            blueprint_data = _to_blueprint_data(data)
            return ExperimentBlueprintCreate(
                **{
                    **blueprint_data,
                    "phases": [BlueprintPhaseCreate(**phase) for phase in blueprint_data["phases"]],
                }
            )
        except BlueprintParsingError:
            raise
        except Exception as e:
            raise BlueprintParsingError(f"Error parsing YAML blueprint: {str(e)}")


class XMLBlueprintParser(BaseBlueprintParser):
    def parse(self, content: str) -> ExperimentBlueprintCreate:
        try:
            root = ET.fromstring(content)
            if root.tag not in {"experiment_blueprint", "blueprint"}:
                raise BlueprintParsingError(
                    "XML root must be <experiment_blueprint> or <blueprint>"
                )

            blueprint_data: Dict[str, Any] = {
                "id": root.findtext("id") or root.findtext("experiment_id"),
                "name": root.findtext("name"),
                "description": root.findtext("description"),
                "phases": [],
            }

            phases_parent = root.find("phases")
            if phases_parent is None:
                raise BlueprintParsingError("XML blueprint must include <phases>")

            for phase_node in phases_parent.findall("phase"):
                modalities = []
                mods_node = phase_node.find("allowed_modalities")
                if mods_node is not None:
                    modalities = [mod.text for mod in mods_node.findall("modality") if mod.text]
                    if not modalities and mods_node.text:
                        modalities = _parse_modalities_field(mods_node.text)

                blueprint_data["phases"].append(
                    {
                        "phase_order": phase_node.findtext("phase_order"),
                        "target_valence": phase_node.findtext("target_valence", "0.0"),
                        "target_arousal": phase_node.findtext("target_arousal", "0.0"),
                        "min_duration_ms": phase_node.findtext("min_duration_ms", "1000"),
                        "max_duration_ms": phase_node.findtext("max_duration_ms", "1000"),
                        "allowed_modalities": modalities,
                    }
                )

            normalized = _to_blueprint_data(blueprint_data)
            return ExperimentBlueprintCreate(
                **{
                    **normalized,
                    "phases": [BlueprintPhaseCreate(**phase) for phase in normalized["phases"]],
                }
            )
        except ET.ParseError as e:
            raise BlueprintParsingError(f"Invalid XML: {str(e)}")
        except BlueprintParsingError:
            raise
        except Exception as e:
            raise BlueprintParsingError(f"Error parsing XML blueprint: {str(e)}")


def get_parser(filename: str) -> BaseQuestionnaireParser:
    lower_name = filename.lower().strip()

    if lower_name.endswith(".json"):
        return JSONQuestionnaireParser()
    elif lower_name.endswith(".xml"):
        return XMLQuestionnaireParser()
    elif lower_name.endswith(".csv"):
        return CSVQuestionnaireParser()
    elif lower_name.endswith(".yaml") or lower_name.endswith(".yml"):
        return YAMLQuestionnaireParser()
    else:
        raise QuestionnaireParsingError(f"Unsupported file format: {filename}")


def get_blueprint_parser(filename: str) -> BaseBlueprintParser:
    lower_name = filename.lower().strip()

    if lower_name.endswith(".json"):
        return JSONBlueprintParser()
    if lower_name.endswith(".xml"):
        return XMLBlueprintParser()
    if lower_name.endswith(".yaml") or lower_name.endswith(".yml"):
        return YAMLBlueprintParser()

    raise BlueprintParsingError(f"Unsupported blueprint file format: {filename}")
