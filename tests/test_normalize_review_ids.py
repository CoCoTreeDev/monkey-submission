from normalize_review_ids import normalize


def test_review_ids_are_sequential_and_keep_source_ids() -> None:
    manifest = {"clips": [{"rows": [{"track_id": 15}, {"track_id": 27}]}]}

    normalize(manifest)

    assert [row["track_id"] for row in manifest["clips"][0]["rows"]] == [1, 2]
    assert [row["source_track_id"] for row in manifest["clips"][0]["rows"]] == [15, 27]
    assert manifest["review_id_scheme"] == "canonical_clip_sequence_v1"
