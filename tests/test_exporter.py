from src.exporter import (
    generate_eprime_tab,
    generate_presentation_tab,
    generate_psychopy_csv,
)


def test_generate_psychopy_csv_contains_header_and_rows():
    playlist = {
        "phases": [
            {
                "phase_id": 1,
                "selected_stimuli": [
                    {"stimulus_id": "s1", "uri": "u1", "distance": 1.23456},
                    {"stimulus_id": "s2", "uri": "u2", "distance": 0.1},
                ],
            }
        ]
    }

    csv_text = generate_psychopy_csv(playlist)

    lines = [line for line in csv_text.strip().splitlines() if line]
    assert (
        lines[0] == "Trial,Phase,Film,Code,Target_Valence,Target_Arousal,"
        "Normalized_Score_Valence,Dist_From_Target,Arousal_Metadata,Duration_ms"
    )
    assert lines[1].startswith("1,1,s1,")
    assert lines[2].startswith("2,1,s2,")


def test_generate_psychopy_csv_handles_empty_playlist():
    csv_text = generate_psychopy_csv({"phases": []})
    lines = [line for line in csv_text.strip().splitlines() if line]
    assert lines == [
        (
            "Trial,Phase,Film,Code,Target_Valence,Target_Arousal,"
            "Normalized_Score_Valence,Dist_From_Target,Arousal_Metadata,Duration_ms"
        ),
    ]


def test_generate_psychopy_csv_with_selected_films_fallback():
    playlist = {
        "phases": [
            {
                "phase_id": "p1",
                "target_valence": 0.5,
                "target_arousal": 0.5,
                "selected_films": [
                    {
                        "film": "f1",
                        "code": "c1",
                        "valence": 0.4,
                        "arousal": 0.6,
                        "distance": 0.15,
                        "duration_seconds": 30,
                    }
                ],
            }
        ]
    }
    csv_text = generate_psychopy_csv(playlist)
    lines = [line for line in csv_text.strip().splitlines() if line]
    assert len(lines) == 2
    # Check headers
    assert "Trial,Phase,Film,Code" in lines[0]
    # Check values mapped from selected_films
    row = lines[1].split(",")
    assert row[2] == "f1"
    assert row[3] == "c1"
    assert row[4] == "0.5"  # target valence
    assert row[6] == "0.4"  # raw valence
    assert row[7] == "0.15"  # distance
    assert row[9] == "30000"  # duration_ms


def test_generate_eprime_tab_with_both_payloads():
    # 1. Test selected_stimuli format
    playlist_stim = {
        "phases": [
            {
                "phase_id": "p1",
                "target_valence": 0.7,
                "target_arousal": 0.3,
                "selected_stimuli": [
                    {
                        "stimulus_id": "s1",
                        "uri": "u1",
                        "duration_ms": 15000,
                    }
                ],
            }
        ]
    }
    tab_text = generate_eprime_tab(playlist_stim)
    lines = [line for line in tab_text.strip().splitlines() if line]
    assert len(lines) == 2
    assert "Weight\tNested\tProcedure\ttrial_id" in lines[0]
    assert "s1\tu1\t0.7\t0.3\t15000" in lines[1]

    # 2. Test selected_films format
    playlist_films = {
        "phases": [
            {
                "phase_id": "p1",
                "target_valence": 0.7,
                "target_arousal": 0.3,
                "selected_films": [
                    {
                        "film": "f1",
                        "uri": "csv://f1",
                        "duration_seconds": 20,
                    }
                ],
            }
        ]
    }
    tab_text2 = generate_eprime_tab(playlist_films)
    lines2 = [line for line in tab_text2.strip().splitlines() if line]
    assert len(lines2) == 2
    assert "f1\tcsv://f1\t0.7\t0.3\t20000" in lines2[1]


def test_generate_presentation_tab_with_both_payloads():
    # 1. Test selected_stimuli format
    playlist_stim = {
        "phases": [
            {
                "phase_id": "p1",
                "target_valence": 0.2,
                "target_arousal": 0.8,
                "selected_stimuli": [
                    {
                        "stimulus_id": "s1",
                        "uri": "u1",
                        "modality": "audio",
                        "duration_ms": 10000,
                    }
                ],
            }
        ]
    }
    tab_text = generate_presentation_tab(playlist_stim)
    lines = [line for line in tab_text.strip().splitlines() if line]
    assert len(lines) == 2
    assert "trial_id\tphase_id\tstimulus_id" in lines[0]
    assert "s1\tu1\taudio\t0.2\t0.8\t10000" in lines[1]

    # 2. Test selected_films format
    playlist_films = {
        "phases": [
            {
                "phase_id": "p2",
                "target_valence": 0.2,
                "target_arousal": 0.8,
                "selected_films": [
                    {
                        "film": "f1",
                        "stimulus_uri": "u2",
                        "modality": "video",
                        "duration_seconds": 15,
                    }
                ],
            }
        ]
    }
    tab_text2 = generate_presentation_tab(playlist_films)
    lines2 = [line for line in tab_text2.strip().splitlines() if line]
    assert len(lines2) == 2
    assert "f1\tu2\tvideo\t0.2\t0.8\t15000" in lines2[1]


def test_generate_playlist_xml_with_selected_films_fallback():
    from src.exporter import generate_playlist_xml

    playlist = {
        "participant_id": "P-XML",
        "experiment_id": "EXP-XML",
        "phases": [
            {
                "phase_id": "p1",
                "target_valence": 0.4,
                "target_arousal": 6.0,
                "target_duration_seconds": 30,
                "selected_films": [
                    {
                        "film": "Film XML",
                        "duration_seconds": 30,
                        "valence": 0.4,
                        "arousal": 6.0,
                    }
                ],
            }
        ],
    }

    xml_text = generate_playlist_xml(playlist)

    assert "<film>Film XML</film>" in xml_text
    assert "<duration_seconds>30</duration_seconds>" in xml_text
