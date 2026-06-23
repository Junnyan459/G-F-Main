from glyph.pipeline import ReadabilityPolicy, enforce_readability_budget, partition_text_for_pages, render_query_guided_pngs
from glyph_ficoco.selection import enforce_protected_indices, select_ordered_indices
from scripts.glyph_llava_ficoco_pipeline import chunk_text_for_ficoco


def test_select_ordered_indices_preserves_neighbors_for_spiky_scores() -> None:
    scores = [0.1, 0.2, 0.95, 0.12, 0.11, 0.88]
    selected = select_ordered_indices(scores, keep_count=3, preserve_neighbors=True, neighbor_radius=1)
    assert selected == [1, 2, 5]


def test_enforce_protected_indices_keeps_anchors() -> None:
    scores = [0.05, 0.91, 0.2, 0.18, 0.89, 0.04]
    selected = select_ordered_indices(scores, keep_count=3, preserve_neighbors=True, neighbor_radius=2)
    protected = [0, 5]
    anchored = enforce_protected_indices(selected, protected, scores, keep_count=3)
    assert anchored == [0, 1, 5]


def test_chunk_text_for_ficoco_uses_overlap_and_boundary_alignment() -> None:
    text = "\n".join(
        [
            'Document 1',
            'alpha',
            'beta',
            'gamma',
            'Document 2',
            'delta',
            'epsilon',
            'zeta',
            'Document 3',
            'eta',
        ]
    )
    chunks = chunk_text_for_ficoco(text, lines_per_chunk=4)
    assert len(chunks) >= 3
    assert chunks[0].splitlines()[0] == 'Document 1'
    assert any(chunk.splitlines()[0] == 'Document 2' for chunk in chunks[1:])
    assert any('Document 3' in chunk for chunk in chunks)


def test_readability_budget_truncates_to_page_budget() -> None:
    unit = 'Document 1\n' + ('alpha ' * 900)
    text = '\n\n'.join([unit, unit, unit, unit])
    policy = ReadabilityPolicy(max_chars_per_page=1000, max_render_pages=2)
    bounded = enforce_readability_budget(text, policy)
    assert len(bounded) <= 2000


def test_partition_text_for_pages_preserves_order() -> None:
    text = '\n\n'.join(['Document 1\na', 'Document 2\nb', 'Document 3\nc'])
    pages = partition_text_for_pages(text, 2)
    assert len(pages) == 2
    assert 'Document 1' in pages[0]
    assert 'Document 3' in pages[1]


def test_render_query_guided_pngs_only_renders_relevant_content(tmp_path) -> None:
    context = '\n\n'.join(
        [
            'Document 1\nCats sleep on the sofa all afternoon.',
            'Document 2\nThe Eiffel Tower is located in Paris, France.',
            'Document 3\nDogs chase tennis balls in the park.',
        ]
    )
    artifacts = render_query_guided_pngs(
        context=context,
        question='Which city is the Eiffel Tower in?',
        output_dir=tmp_path,
        unique_id='query_guided_test',
        retrieval_max_units=1,
        retrieval_neighbor_window=0,
    )
    assert artifacts.image_paths
    assert 'Eiffel Tower' in artifacts.selected_text
    assert 'Dogs chase tennis balls' not in artifacts.selected_text
    assert any('Paris' in page for page in artifacts.page_texts)
