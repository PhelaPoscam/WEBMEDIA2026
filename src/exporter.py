import csv
import io
import json
import xml.etree.ElementTree as ET

from src._yaml import get_yaml

yaml = get_yaml()


def generate_psychopy_csv(playlist_dict: dict) -> str:
    """
    Parses the JSON playlist output from the Target Engine and flattens
    it into a trial-by-trial CSV structure compatible with PsychoPy loop
    handlers.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Base columns
    writer.writerow(
        [
            "Trial",
            "Phase",
            "Film",
            "Code",
            "Target_Valence",
            "Target_Arousal",
            "Normalized_Score_Valence",
            "Dist_From_Target",
            "Arousal_Metadata",
            "Duration_ms",
        ]
    )

    trial_counter = 1
    phases = playlist_dict.get("phases", [])

    for phase in phases:
        phase_id = phase.get("phase_id", 0)
        t_val = phase.get("target_valence", 0.0)
        t_aro = phase.get("target_arousal", 0.0)
        stimuli = phase.get("selected_stimuli")
        if stimuli is None:
            stimuli = phase.get("selected_films", [])

        for stim in stimuli:
            duration_ms = stim.get("duration_ms")
            if duration_ms is None and "duration_seconds" in stim:
                duration_ms = int(stim["duration_seconds"]) * 1000

            writer.writerow(
                [
                    trial_counter,
                    phase_id,
                    stim.get("stimulus_id") or stim.get("film", ""),
                    stim.get("stimulus_code") or stim.get("code") or "N/A",
                    round(t_val, 3),
                    round(t_aro, 3),
                    round(stim.get("raw_valence") or stim.get("valence", 0.0), 3),
                    round(stim.get("distance", 0.0), 3),
                    round(stim.get("raw_arousal") or stim.get("arousal", 0.0), 3),
                    duration_ms if duration_ms is not None else 60000,
                ]
            )
            trial_counter += 1

    return output.getvalue()


def generate_eprime_tab(playlist_dict: dict) -> str:
    """
    Generates a Tab-Delimited text file compatible with E-Prime List objects.
    """
    output = io.StringIO()
    headers = [
        "Weight",
        "Nested",
        "Procedure",
        "trial_id",
        "phase_id",
        "stimulus_id",
        "stimulus_uri",
        "target_valence",
        "target_arousal",
        "duration_ms",
    ]
    output.write("\t".join(headers) + "\n")

    trial_counter = 1
    phases = playlist_dict.get("phases", [])

    for phase in phases:
        phase_id = phase.get("phase_id", 0)
        t_val = phase.get("target_valence", 0.0)
        t_aro = phase.get("target_arousal", 0.0)
        stimuli = phase.get("selected_stimuli")
        if stimuli is None:
            stimuli = phase.get("selected_films", [])

        for stim in stimuli:
            duration_ms = stim.get("duration_ms")
            if duration_ms is None and "duration_seconds" in stim:
                duration_ms = int(stim["duration_seconds"]) * 1000

            row = [
                "1",
                "",
                "TrialProc",
                str(trial_counter),
                str(phase_id),
                stim.get("stimulus_id") or stim.get("film", ""),
                stim.get("uri") or stim.get("stimulus_uri") or "",
                str(round(t_val, 3)),
                str(round(t_aro, 3)),
                str(duration_ms if duration_ms is not None else 60000),
            ]
            output.write("\t".join(row) + "\n")
            trial_counter += 1

    return output.getvalue()


def generate_presentation_tab(playlist_dict: dict) -> str:
    """
    Generates a Tab-Delimited text file for NBS Presentation scenario logs.
    """
    output = io.StringIO()
    headers = [
        "trial_id",
        "phase_id",
        "stimulus_id",
        "stimulus_uri",
        "modality",
        "target_valence",
        "target_arousal",
        "duration_ms",
    ]
    output.write("\t".join(headers) + "\n")

    trial_counter = 1
    phases = playlist_dict.get("phases", [])

    for phase in phases:
        phase_id = phase.get("phase_id", 0)
        t_val = phase.get("target_valence", 0.0)
        t_aro = phase.get("target_arousal", 0.0)
        stimuli = phase.get("selected_stimuli")
        if stimuli is None:
            stimuli = phase.get("selected_films", [])

        for stim in stimuli:
            duration_ms = stim.get("duration_ms")
            if duration_ms is None and "duration_seconds" in stim:
                duration_ms = int(stim["duration_seconds"]) * 1000

            row = [
                str(trial_counter),
                str(phase_id),
                stim.get("stimulus_id") or stim.get("film", ""),
                stim.get("uri") or stim.get("stimulus_uri") or "",
                stim.get("modality", "unknown"),
                str(round(t_val, 3)),
                str(round(t_aro, 3)),
                str(duration_ms if duration_ms is not None else 60000),
            ]
            output.write("\t".join(row) + "\n")
            trial_counter += 1

    return output.getvalue()


def generate_playlist_json(playlist_dict: dict) -> str:
    """Return a JSON-formatted string for the playlist."""
    return json.dumps(playlist_dict, indent=2)


def generate_playlist_yaml(playlist_dict: dict) -> str:
    """Return a YAML-formatted string for the playlist."""
    if yaml is None:
        raise RuntimeError("YAML support is unavailable because PyYAML is not installed")
    return yaml.safe_dump(playlist_dict, sort_keys=False, allow_unicode=False)


def generate_playlist_xml(playlist_dict: dict) -> str:
    """Return an XML-formatted string for the playlist."""
    root = ET.Element("session_playlist")
    ET.SubElement(root, "experiment_id").text = str(playlist_dict.get("experiment_id", ""))
    ET.SubElement(root, "participant_id").text = str(playlist_dict.get("participant_id", ""))

    phases_el = ET.SubElement(root, "phases")
    for phase in playlist_dict.get("phases", []):
        phase_el = ET.SubElement(phases_el, "phase")
        ET.SubElement(phase_el, "phase_id").text = str(phase.get("phase_id", ""))
        ET.SubElement(phase_el, "target_valence").text = str(phase.get("target_valence", ""))
        ET.SubElement(phase_el, "target_arousal").text = str(phase.get("target_arousal", ""))
        ET.SubElement(phase_el, "target_duration_seconds").text = str(
            phase.get("target_duration_seconds", "")
        )

        stimuli_el = ET.SubElement(phase_el, "selected_stimuli")
        stimuli = phase.get("selected_stimuli")
        if stimuli is None:
            stimuli = phase.get("selected_films", [])

        for stim in stimuli:
            stim_el = ET.SubElement(stimuli_el, "stimulus")
            for key, val in stim.items():
                ET.SubElement(stim_el, key).text = str(val)

    ET.indent(root, space="  ", level=0)
    return ET.tostring(root, encoding="unicode")


def generate_participant_profile_payload(profile_dict: dict) -> dict:
    """Normalize participant profile payload for export formats."""
    return {
        "participant_id": profile_dict.get("id", ""),
        "baseline_valence": profile_dict.get("baseline_valence", 0.0),
        "baseline_arousal": profile_dict.get("baseline_arousal", 0.0),
        "phobias": profile_dict.get("phobias", []) or [],
        "familiar_media": profile_dict.get("familiar_media", []) or [],
        "preferences": profile_dict.get("preferences", {}) or {},
        "is_excluded": profile_dict.get("is_excluded", False),
        "exclusion_reason": profile_dict.get("exclusion_reason"),
    }


def generate_participant_profile_xml(profile_dict: dict) -> str:
    """Generate questionnaire-compatible XML for a participant profile."""
    payload = generate_participant_profile_payload(profile_dict)
    root = ET.Element("questionnaire")

    ET.SubElement(root, "participant_id").text = str(payload["participant_id"])
    ET.SubElement(root, "baseline_valence").text = str(payload["baseline_valence"])
    ET.SubElement(root, "baseline_arousal").text = str(payload["baseline_arousal"])

    phobias_el = ET.SubElement(root, "phobias")
    for phobia in payload["phobias"]:
        ET.SubElement(phobias_el, "phobia").text = str(phobia)

    familiar_el = ET.SubElement(root, "familiar_media")
    for item in payload["familiar_media"]:
        ET.SubElement(familiar_el, "item").text = str(item)

    prefs_el = ET.SubElement(root, "preferences")
    for key, val in payload["preferences"].items():
        pref_el = ET.SubElement(prefs_el, "preference")
        pref_el.set("id", str(key))
        pref_el.text = str(val)

    if payload["is_excluded"]:
        ET.SubElement(root, "is_excluded").text = "true"
    if payload["exclusion_reason"]:
        ET.SubElement(root, "exclusion_reason").text = str(payload["exclusion_reason"])

    ET.indent(root, space="  ", level=0)
    return ET.tostring(root, encoding="unicode")


def generate_participant_profile_yaml(profile_dict: dict) -> str:
    """Generate YAML for a participant profile payload."""
    if yaml is None:
        raise RuntimeError("YAML support is unavailable because PyYAML is not installed")
    payload = generate_participant_profile_payload(profile_dict)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def generate_participant_profile_json(profile_dict: dict) -> str:
    """Generate JSON text for a participant profile payload."""
    payload = generate_participant_profile_payload(profile_dict)
    return json.dumps(payload, indent=2)


def generate_blueprint_payload(blueprint_dict: dict) -> dict:
    """Normalize experiment blueprint payload for export formats."""
    phases = blueprint_dict.get("phases", []) or []
    normalized_phases = []
    for phase in phases:
        normalized_phases.append(
            {
                "phase_order": phase.get("phase_order"),
                "target_valence": phase.get("target_valence"),
                "target_arousal": phase.get("target_arousal"),
                "min_duration_ms": phase.get("min_duration_ms"),
                "max_duration_ms": phase.get("max_duration_ms"),
                "allowed_modalities": (phase.get("allowed_modalities", []) or []),
            }
        )

    return {
        "id": blueprint_dict.get("id", ""),
        "name": blueprint_dict.get("name", ""),
        "description": blueprint_dict.get("description"),
        "phases": sorted(
            normalized_phases,
            key=lambda item: int(item.get("phase_order") or 0),
        ),
    }


def generate_blueprint_json(blueprint_dict: dict) -> str:
    """Generate JSON text for an experiment blueprint payload."""
    payload = generate_blueprint_payload(blueprint_dict)
    return json.dumps(payload, indent=2)


def generate_blueprint_yaml(blueprint_dict: dict) -> str:
    """Generate YAML text for an experiment blueprint payload."""
    if yaml is None:
        raise RuntimeError("YAML support is unavailable because PyYAML is not installed")
    payload = generate_blueprint_payload(blueprint_dict)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def generate_blueprint_xml(blueprint_dict: dict) -> str:
    """Generate XML text for an experiment blueprint payload."""
    payload = generate_blueprint_payload(blueprint_dict)
    root = ET.Element("experiment_blueprint")

    ET.SubElement(root, "id").text = str(payload["id"])
    ET.SubElement(root, "name").text = str(payload["name"])
    if payload.get("description") is not None:
        ET.SubElement(root, "description").text = str(payload["description"])

    phases_el = ET.SubElement(root, "phases")
    for phase in payload["phases"]:
        phase_el = ET.SubElement(phases_el, "phase")
        ET.SubElement(phase_el, "phase_order").text = str(phase.get("phase_order", ""))
        ET.SubElement(phase_el, "target_valence").text = str(phase.get("target_valence", ""))
        ET.SubElement(phase_el, "target_arousal").text = str(phase.get("target_arousal", ""))
        ET.SubElement(phase_el, "min_duration_ms").text = str(phase.get("min_duration_ms", ""))
        ET.SubElement(phase_el, "max_duration_ms").text = str(phase.get("max_duration_ms", ""))

        mods_el = ET.SubElement(phase_el, "allowed_modalities")
        for modality in phase.get("allowed_modalities", []):
            ET.SubElement(mods_el, "modality").text = str(modality)

    ET.indent(root, space="  ", level=0)
    return ET.tostring(root, encoding="unicode")


def generate_transparency_report(playlist_dict: dict) -> str:
    """
    Generates a beautifully structured human-readable Markdown transparency report
    detailing the session generation run metadata, exclusions, and phase selection details.
    """
    report = []
    report.append("# ASA Session Playlist Transparency Report\n")

    # Run Metadata
    meta = playlist_dict.get("transparency_report", {}).get("run_metadata", {})
    report.append("## 1. Run Metadata")
    report.append(f"- **Participant ID**: {playlist_dict.get('participant_id', 'N/A')}")
    report.append(f"- **Experiment ID**: {playlist_dict.get('experiment_id', 'N/A')}")
    report.append(f"- **Matching Strategy**: {meta.get('matching_strategy', 'N/A')}")
    report.append(f"- **Valence Model**: {playlist_dict.get('valence_model', 'N/A')}")
    report.append(f"- **Valence Column**: {playlist_dict.get('valence_column', 'N/A')}")
    report.append(f"- **Total Input Pool Size**: {meta.get('input_pool_size', 'N/A')}")
    post_pool_size = playlist_dict.get(
        "post_familiarity_pool_size", playlist_dict.get("unassigned_pool_size", "N/A")
    )
    report.append(f"- **Final Pool Size (Post-Exclusions)**: {post_pool_size}")
    report.append("")

    # Exclusions Log
    exclusions = playlist_dict.get("transparency_report", {}).get("exclusions", [])
    report.append("## 2. Exclusions Log")
    if not exclusions:
        report.append("No stimuli were excluded during pre-filtering or phase constraint checks.")
    else:
        report.append("| Stimulus ID | Exclusion Reason |")
        report.append("| :--- | :--- |")
        for exc in exclusions:
            report.append(f"| {exc.get('stimulus_id')} | {exc.get('reason')} |")
    report.append("")

    # Phase-by-Phase Selection
    report.append("## 3. Phase-by-Phase Selection")
    phases = playlist_dict.get("transparency_report", {}).get("phases", [])
    for phase in phases:
        phase_id = phase.get("phase_id")
        report.append(f"### Phase: {phase_id}")
        report.append(f"- **Target Valence**: {phase.get('target_valence')}")
        report.append(f"- **Target Arousal**: {phase.get('target_arousal')}")
        achieved_dur = phase.get("achieved_duration_seconds")
        target_dur = phase.get("target_duration_seconds")
        report.append(f"- **Achieved Duration**: {achieved_dur}s (Target: {target_dur}s)")
        report.append(f"- **Duration Gap**: {phase.get('duration_gap_seconds')}s")
        report.append(f"- **Phase Candidate Pool Size**: {phase.get('candidate_pool_size')}")
        report.append("")

        report.append("#### Selected Clips")
        selections = phase.get("selections", [])
        if not selections:
            report.append("No clips were selected for this phase.")
        else:
            report.append("| Film / Stimulus | Distance | Duration | Selection Reason |")
            report.append("| :--- | :--- | :--- | :--- |")
            for sel in selections:
                dur = sel.get("duration_seconds", sel.get("duration_ms", 0) // 1000)
                reason = sel.get("selection_reason", "N/A")
                reason_clean = reason.replace("_", " ").title()
                stim_id = sel.get("stimulus_id")
                dist = sel.get("distance")
                report.append(f"| {stim_id} | {dist:.6f} | {dur}s | {reason_clean} |")
        report.append("")

        rejected = phase.get("rejected_candidates", [])
        if rejected:
            report.append("#### Near-Candidates (Rejected/Not Selected)")
            report.append("| Film / Stimulus | Distance | Duration |")
            report.append("| :--- | :--- | :--- |")
            for rej in rejected:
                rej_dur = rej.get("duration_seconds", rej.get("duration_ms", 0) // 1000)
                report.append(
                    f"| {rej.get('stimulus_id')} | {rej.get('distance'):.6f} | {rej_dur}s |"
                )
        report.append("\n---")

    return "\n".join(report)
