from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fitz

from patra_mineru import llm_client
from patra_mineru.ir_builder import _item_text, build_document_ir
from patra_mineru.llm_client import build_chat_request, extract_json_content, response_content
from patra_mineru.mineru_runner import build_mineru_command, build_mineru_env
from patra_mineru.models import to_jsonable
from patra_mineru.models import Block, RichSpan
from patra_mineru.payload import (
    build_mock_response,
    build_payload,
    continues_into_next,
    restored_response_block,
    split_payload,
    untranslated_source_sentences,
)
from patra_mineru.style import _rich_text_is_usable, block_rect, merge_rich_spans
from patra_mineru.protect import protect_text, restore_text
from patra_mineru.reconstruct import render_pdf, rich_spans_to_html


class PipelineTests(unittest.TestCase):
    def test_protect_and_restore(self) -> None:
        text, mapping = protect_text("The attention score is $qk^T$.", ["attention"])
        self.assertIn("KEEP_TERM", text)
        self.assertIn("KEEP_EQ", text)
        self.assertEqual(restore_text(text, mapping), "The attention score is $qk^T$.")

    def test_merges_adjacent_same_style_rich_spans(self) -> None:
        spans = [
            (10.0, 10.0, 25.0, RichSpan(text="The", font_size=9.0, font_name="Times")),
            (10.0, 28.0, 70.0, RichSpan(text="attention", font_size=9.0, font_name="Times")),
            (10.0, 75.0, 110.0, RichSpan(text="score", italic=True, font_size=9.0, font_name="Times-Italic")),
        ]
        merged = merge_rich_spans(spans)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].text, "The attention")
        self.assertFalse(merged[0].italic)
        self.assertEqual(merged[1].text, " score")
        self.assertTrue(merged[1].italic)

    def test_rich_text_validation_rejects_neighboring_extra_text(self) -> None:
        source = "TTX Cost Analysis. The training phase is efficient."
        good = [RichSpan(text="TTX Cost Analysis. The training phase is efficient.")]
        bad = [RichSpan(text="between measurement overhead and performance gain. TTX Cost Analysis. The training phase is efficient.")]
        self.assertTrue(_rich_text_is_usable(good, source))
        self.assertFalse(_rich_text_is_usable(bad, source))

    def test_builds_openai_compatible_chat_request(self) -> None:
        payload = {
            "system_prompt": "Return JSON only.",
            "task": "Translate to Korean.",
            "response_schema": {"blocks": []},
            "blocks": [{"id": "b_1", "text": "Hello"}],
        }
        request = build_chat_request(payload, model="test-model", temperature=0.0, json_mode=True)
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertIn("Hello", request["messages"][1]["content"])
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertIn("output_format_example", request["messages"][1]["content"])
        self.assertNotIn("response_schema", request["messages"][1]["content"])

    def test_extracts_json_content_from_fenced_model_output(self) -> None:
        parsed = extract_json_content("```json\n{\"blocks\":[{\"id\":\"b_1\",\"text\":\"안녕\"}]}\n```")
        self.assertEqual(parsed["blocks"][0]["id"], "b_1")

    def test_extracts_json_content_after_reasoning_block(self) -> None:
        content = (
            "<think>\nI should return {\"blocks\": ...} with the ids kept.\n</think>\n"
            "{\"blocks\":[{\"id\":\"b_1\",\"text\":\"안녕\"}]}"
        )
        parsed = extract_json_content(content)
        self.assertEqual(parsed["blocks"][0]["text"], "안녕")

    def test_extracts_json_content_nested_under_echoed_envelope_key(self) -> None:
        content = json.dumps(
            {
                "task": "Translate the academic-paper text to Korean.",
                "response_schema": {"blocks": [{"id": "b_1", "text": "안녕"}]},
            }
        )
        parsed = extract_json_content(content)
        self.assertEqual(parsed["blocks"][0]["id"], "b_1")

    def test_rejects_model_json_without_any_blocks_list(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_content("{\"task\":\"Translate\",\"note\":\"no blocks here\"}")

    def test_salvages_blocks_when_the_envelope_is_closed_after_every_block(self) -> None:
        # Qwen3 repeats the closing "]}" after each block, so json.loads stops after
        # the first object with "Extra data". Each block is still valid on its own.
        content = (
            "<think>\nplanning\n</think>\n"
            '{"blocks":[{"id":"b_1","text":"첫째","rich_text":[{"text":"첫째","bold":true}]}]}'
            ',{"id":"b_2","text":"둘째"}]}'
            ',{"id":"b_3","text":"셋째"}]}'
        )
        notes: list[str] = []
        parsed = extract_json_content(content, notes=notes)
        self.assertEqual([b["id"] for b in parsed["blocks"]], ["b_1", "b_2", "b_3"])
        self.assertEqual(parsed["blocks"][0]["rich_text"][0]["text"], "첫째")
        self.assertTrue(any("recovered 3" in note for note in notes))

    def test_salvage_does_not_hide_a_response_with_no_blocks(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            extract_json_content('{"blocks":[{"id":"b_1","text":"첫')

    def test_reports_a_completion_the_server_cut_short(self) -> None:
        raw = {
            "choices": [{"finish_reason": "length", "message": {"content": "{\"blocks\":[{\"id\":\"b_1\",\"text\":\"안"}}],
            "usage": {"prompt_tokens": 37386, "completion_tokens": 3574, "total_tokens": 40960},
        }
        with self.assertRaises(RuntimeError) as caught:
            response_content(raw)
        self.assertIn("finish_reason=length", str(caught.exception))
        self.assertIn("37386", str(caught.exception))

    def test_splits_a_payload_into_request_sized_batches(self) -> None:
        payload = {
            "system_prompt": "Return strict JSON only.",
            "task": "Translate to Korean.",
            "response_schema": {"blocks": []},
            "blocks": [{"id": f"b_{i}", "text": "x" * 400} for i in range(10)],
        }
        batches = split_payload(payload, max_blocks=3, max_chars=0)
        self.assertEqual([len(batch["blocks"]) for batch in batches], [3, 3, 3, 1])
        self.assertEqual(batches[0]["system_prompt"], payload["system_prompt"])
        self.assertEqual(batches[0]["task"], payload["task"])
        self.assertEqual(
            [block["id"] for batch in batches for block in batch["blocks"]],
            [block["id"] for block in payload["blocks"]],
        )

        by_chars = split_payload(payload, max_blocks=0, max_chars=900)
        self.assertTrue(all(len(batch["blocks"]) == 2 for batch in by_chars))
        # A block larger than the budget is never dropped or cut in half.
        big = {"blocks": [{"id": "b_0", "text": "x" * 5000}, {"id": "b_1", "text": "y"}]}
        self.assertEqual([len(b["blocks"]) for b in split_payload(big, max_blocks=0, max_chars=100)], [1, 1])
        self.assertEqual(len(split_payload(payload, max_blocks=0, max_chars=0)), 1)

    def test_translate_payload_merges_the_batched_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload_path = root / "llm_payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "system_prompt": "Return strict JSON only.",
                        "task": "Translate to Korean.",
                        "response_schema": {"blocks": []},
                        "blocks": [{"id": f"b_{i}", "text": f"sentence {i}"} for i in range(5)],
                    }
                ),
                encoding="utf-8",
            )
            sent: list[dict] = []

            def fake_call(*, base_url, api_key, request_body, timeout=300.0):
                blocks = json.loads(request_body["messages"][1]["content"])["blocks"]
                sent.append(request_body)
                content = json.dumps({"blocks": [{"id": b["id"], "text": "번역 " + b["text"]} for b in blocks]})
                return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

            raw_path = root / "raw_response.json"
            with mock.patch.object(llm_client, "call_openai_compatible", fake_call):
                result = llm_client.translate_payload(
                    payload_path=payload_path,
                    output_path=root / "response.json",
                    base_url="http://localhost:8000/v1",
                    model="test-model",
                    batch_blocks=2,
                    batch_chars=0,
                    raw_response_out=raw_path,
                )

            self.assertEqual(len(sent), 3)
            self.assertEqual([b["id"] for b in result["blocks"]], [f"b_{i}" for i in range(5)])
            self.assertEqual(json.loads((root / "response.json").read_text(encoding="utf-8")), result)
            # One completion per batch is kept, so a later failure loses nothing.
            self.assertEqual(len(json.loads(raw_path.read_text(encoding="utf-8"))), 3)

    def test_translate_payload_warns_about_blocks_a_batch_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload_path = root / "llm_payload.json"
            payload_path.write_text(
                json.dumps({"blocks": [{"id": f"b_{i}", "text": f"sentence {i}"} for i in range(4)]}),
                encoding="utf-8",
            )

            def fake_call(*, base_url, api_key, request_body, timeout=300.0):
                blocks = json.loads(request_body["messages"][1]["content"])["blocks"]
                kept = [{"id": b["id"], "text": "번역"} for b in blocks if b["id"] != "b_3"]
                return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"blocks": kept})}}]}

            messages: list[str] = []
            with mock.patch.object(llm_client, "call_openai_compatible", fake_call):
                result = llm_client.translate_payload(
                    payload_path=payload_path,
                    output_path=root / "response.json",
                    base_url="http://localhost:8000/v1",
                    model="test-model",
                    batch_blocks=2,
                    batch_chars=0,
                    warn=messages.append,
                )

            self.assertEqual([b["id"] for b in result["blocks"]], ["b_0", "b_1", "b_2"])
            self.assertEqual(len(messages), 1)
            self.assertIn("b_3", messages[0])

    def test_batches_do_not_cut_between_continued_blocks(self) -> None:
        payload = {
            "blocks": [
                {"id": "b_0", "text": "one"},
                {"id": "b_1", "text": "two", "continues_next": True},
                {"id": "b_2", "text": "three", "continues_previous": True},
                {"id": "b_3", "text": "four"},
            ]
        }
        batches = split_payload(payload, max_blocks=2, max_chars=0)
        self.assertEqual([[b["id"] for b in batch["blocks"]] for batch in batches], [["b_0"], ["b_1", "b_2"], ["b_3"]])

    def test_discards_rich_text_that_echoes_the_source_block(self) -> None:
        block = Block(
            id="b_1",
            type="text",
            page_idx=0,
            bbox=[0, 0, 100, 100],
            text="TTX combines input shapes and tuning parameters to predict latency.",
            rich_text=[RichSpan(text="TTX combines input shapes and tuning parameters to predict latency.", bold=True)],
        )
        response_block = {
            "id": "b_1",
            "text": "TTX는 입력 형상과 튜닝 매개변수를 결합하여 지연을 예측합니다.",
            "rich_text": [{"text": "TTX combines input shapes and tuning parameters to predict latency."}],
        }
        notes: list[str] = []
        text, rich = restored_response_block(block, response_block, notes=notes)
        self.assertEqual(len(rich), 1)
        self.assertEqual(rich[0].text, text)
        self.assertNotIn("combines", rich[0].text)
        self.assertTrue(rich[0].bold, "fallback should inherit the source span style")
        self.assertTrue(any("untransformed" in note for note in notes))

    def test_keeps_rich_text_that_matches_the_response_text(self) -> None:
        block = Block(
            id="b_1",
            type="text",
            page_idx=0,
            bbox=[0, 0, 100, 100],
            text="The attention score is important.",
            rich_text=[RichSpan(text="The attention score is important.")],
        )
        response_block = {
            "id": "b_1",
            "text": "어텐션 점수는 중요합니다.",
            "rich_text": [{"text": "어텐션 점수는 ", "bold": False}, {"text": "중요합니다.", "bold": True}],
        }
        notes: list[str] = []
        _, rich = restored_response_block(block, response_block, notes=notes)
        self.assertEqual(len(rich), 2)
        self.assertTrue(rich[1].bold)
        self.assertEqual(notes, [])

    def test_flags_whole_source_sentences_left_untransformed(self) -> None:
        source = (
            "We conduct the experiments with comprehensive operators. "
            "The experiments are conducted on Nvidia V100S-PCIE-32GB and AMD Instinct MI250 GPUs."
        )
        rendered = (
            "우리는 다양한 연산자로 실험을 수행합니다. "
            "The experiments are conducted on Nvidia V100S-PCIE-32GB and AMD Instinct MI250 GPUs."
        )
        survivors = untranslated_source_sentences(source, rendered)
        self.assertEqual(len(survivors), 1)
        self.assertTrue(survivors[0].startswith("The experiments are conducted"))

    def test_preserved_terms_and_citations_are_not_flagged(self) -> None:
        source = "Frameworks such as TensorFlow [16], PyTorch [17], and vLLM [18] are widely used."
        rendered = "TensorFlow [16], PyTorch [17], vLLM [18]과 같은 프레임워크가 널리 사용됩니다."
        self.assertEqual(untranslated_source_sentences(source, rendered), [])

    def test_headings_are_kept_in_the_source_language_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "paper.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=400)
            page.insert_text((30, 50), "III. Evaluation", fontsize=14)
            page.insert_text((30, 90), "The attention score is important.", fontsize=10)
            doc.save(pdf_path)
            doc.close()

            mineru_dir = root / "mineru"
            mineru_dir.mkdir()
            content = [
                {"type": "text", "text": "III. Evaluation", "text_level": 1, "bbox": [100, 100, 700, 150], "page_idx": 0},
                {"type": "text", "text": "The attention score is important.", "bbox": [100, 200, 900, 240], "page_idx": 0},
            ]
            (mineru_dir / "paper_content_list.json").write_text(json.dumps(content), encoding="utf-8")

            ir = build_document_ir(pdf_path, mineru_dir)
            self.assertTrue(ir.blocks[0].excluded, "headings should not be sent by default")
            self.assertFalse(ir.blocks[1].excluded)
            self.assertEqual([b["id"] for b in build_payload(ir)["blocks"]], ["b_00001"])

            translated = build_document_ir(pdf_path, mineru_dir, translate_headings=True)
            self.assertFalse(translated.blocks[0].excluded)
            self.assertEqual(len(build_payload(translated)["blocks"]), 2)

    def test_mineru_list_blocks_carry_their_items(self) -> None:
        # MinerU 3.x emits list blocks with list_items and no text field.
        item = {
            "type": "list",
            "sub_type": "text",
            "list_items": ["2) Kernel Performance Prediction: We build a predictor.", "3) Model-Guided Tuning: Our system takes input shapes."],
            "bbox": [91, 441, 483, 741],
            "page_idx": 1,
        }
        text = _item_text(item)
        self.assertIn("Kernel Performance Prediction", text)
        self.assertIn("Model-Guided Tuning", text)
        self.assertEqual(text.count("\n"), 1)
        self.assertEqual(_item_text({"type": "list", "list_items": [], "bbox": [0, 0, 1, 1]}), "")

    def test_list_items_stay_on_separate_rendered_lines(self) -> None:
        markup = rich_spans_to_html([RichSpan(text="첫 번째 항목\n두 번째 항목")], "", preserve_newlines=True)
        self.assertIn("<br/>", markup)

    def test_flags_sentences_split_across_blocks(self) -> None:
        self.assertTrue(continues_into_next("TTX combines input shapes and IR-", "level features to predict latency."))
        self.assertFalse(continues_into_next("TTX predicts kernel latency.", "The experiments run on four GPUs."))
        self.assertFalse(continues_into_next("Results are shown in Figure 4.", "however, the trend holds."))
        self.assertFalse(continues_into_next("A trailing fragment", ""))

    def test_payload_marks_continued_block_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "paper.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=400)
            page.insert_text((30, 50), "TTX combines input shapes and IR-", fontsize=10)
            page.insert_text((30, 90), "level features to predict latency.", fontsize=10)
            page.insert_text((30, 130), "A separate finished sentence.", fontsize=10)
            doc.save(pdf_path)
            doc.close()

            mineru_dir = root / "mineru"
            mineru_dir.mkdir()
            content = [
                {"type": "text", "text": "TTX combines input shapes and IR-", "bbox": [100, 100, 900, 140], "page_idx": 0},
                {"type": "text", "text": "level features to predict latency.", "bbox": [100, 200, 900, 240], "page_idx": 0},
                {"type": "text", "text": "A separate finished sentence.", "bbox": [100, 300, 900, 340], "page_idx": 0},
            ]
            (mineru_dir / "paper_content_list.json").write_text(json.dumps(content), encoding="utf-8")

            payload = build_payload(build_document_ir(pdf_path, mineru_dir))
            first, second, third = payload["blocks"]
            self.assertTrue(first.get("continues_next"))
            self.assertTrue(second.get("continues_previous"))
            self.assertNotIn("continues_next", second)
            self.assertNotIn("continues_previous", third)

    def _split_paragraph_fixture(self, root: Path) -> tuple[Path, Path]:
        """A paragraph a page break split, the way MinerU reports it.

        The full text sits on the first block; the leftover rectangle on the next
        page comes back as an empty text block.
        """
        head = "The kernel performance predictor takes tuning parameters"
        tail = "and input shape features and predicts the execution time."

        pdf_path = root / "paper.pdf"
        doc = fitz.open()
        first = doc.new_page(width=300, height=400)
        first.insert_text((30, 300), head, fontsize=9)
        second = doc.new_page(width=300, height=400)
        second.insert_text((30, 60), tail, fontsize=9)
        second.insert_text((30, 200), "An unrelated later paragraph body.", fontsize=9)
        doc.save(pdf_path)
        doc.close()

        mineru_dir = root / "mineru"
        mineru_dir.mkdir()
        content = [
            {"type": "text", "text": f"{head} {tail}", "bbox": [80, 730, 900, 765], "page_idx": 0},
            {"type": "text", "text": "", "bbox": [80, 125, 900, 200], "page_idx": 1},
            {"type": "text", "text": "", "bbox": [80, 480, 900, 520], "page_idx": 1},
        ]
        (mineru_dir / "paper_content_list.json").write_text(json.dumps(content), encoding="utf-8")
        return pdf_path, mineru_dir

    def test_page_break_leftover_becomes_a_second_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path, mineru_dir = self._split_paragraph_fixture(Path(tmp))
            ir = build_document_ir(pdf_path, mineru_dir)
            owner, leftover, unrelated = ir.blocks

            self.assertEqual([r.page_idx for r in owner.regions], [0, 1])
            self.assertEqual(owner.regions[1].bbox, [80, 125, 900, 200])
            self.assertEqual(leftover.continuation_of, owner.id)
            # An empty block whose region holds something else is left alone.
            self.assertIsNone(unrelated.continuation_of)
            self.assertEqual(unrelated.regions, [])
            # Styled spans are read from both rectangles, not just the first.
            self.assertIn("execution time", "".join(span.text for span in owner.rich_text))

    def test_render_flows_a_split_block_through_all_its_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path, mineru_dir = self._split_paragraph_fixture(root)
            ir = build_document_ir(pdf_path, mineru_dir)
            translated = (
                "커널 성능 예측기는 튜닝 파라미터와 입력 형태 특징을 입력으로 받아 "
                "실행 시간을 추정한다. 배포 단계에서 예측기는 처음 보는 입력 형태로도 일반화된다."
            )
            response = {"blocks": [{"id": ir.blocks[0].id, "text": translated}]}

            output_pdf = root / "rendered.pdf"
            warnings = render_pdf(pdf_path, ir, response, output_pdf)
            self.assertEqual(warnings, [])

            rendered = fitz.open(output_pdf)
            try:
                second = rendered[1]
                drawn = second.get_text("text", clip=block_rect(second, [80, 125, 900, 200]))
            finally:
                rendered.close()
            # The tail of the translation has to land in the leftover rectangle;
            # before this it stayed in the first box and the source text showed here.
            self.assertIn("일반화된다", drawn)

    def test_source_line_wrapping_does_not_become_forced_breaks(self) -> None:
        spans = [RichSpan(text="TTX combines input shapes and IR-\nlevel features\nto predict latency.")]
        markup = rich_spans_to_html(spans, "")
        self.assertNotIn("<br/>", markup)
        self.assertIn("IR-level features to predict latency.", markup)
        code = rich_spans_to_html([RichSpan(text="a = 1\nb = 2")], "", preserve_newlines=True)
        self.assertIn("<br/>", code)

    def test_builds_mineru_command_for_a_running_vlm_server(self) -> None:
        command = build_mineru_command(
            "paper.pdf", "out", backend="vlm-http-client", server_url="http://127.0.0.1:30000"
        )
        self.assertEqual(command[:2], ["mineru", "-p"])
        self.assertIn("-u", command)
        self.assertEqual(command[command.index("-u") + 1], "http://127.0.0.1:30000")
        self.assertEqual(command[command.index("-b") + 1], "vlm-http-client")
        self.assertNotIn("-m", command, "method is only meaningful for pipeline and hybrid backends")

    def test_api_url_and_server_url_compose(self) -> None:
        # A long-lived mineru-api handles orchestration while inference goes to a
        # separate VLM server, so both flags must survive onto the command line.
        command = build_mineru_command(
            "paper.pdf",
            "out",
            backend="vlm-http-client",
            server_url="http://127.0.0.1:50020",
            api_url="http://127.0.0.1:50021",
            lang="en",
        )
        self.assertEqual(command[command.index("--api-url") + 1], "http://127.0.0.1:50021")
        self.assertEqual(command[command.index("-u") + 1], "http://127.0.0.1:50020")
        self.assertEqual(command[command.index("-l") + 1], "en")

    def test_http_client_backend_requires_a_server_url(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_mineru_command("paper.pdf", "out", backend="hybrid-http-client")
        self.assertIn("--server-url", str(ctx.exception))

    def test_device_selection_controls_the_subprocess_environment(self) -> None:
        cpu = build_mineru_env("cpu")
        self.assertEqual(cpu["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(cpu["MINERU_DEVICE_MODE"], "cpu")

        auto = build_mineru_env("auto")
        self.assertNotIn("MINERU_DEVICE_MODE", auto)
        self.assertEqual(auto.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))

        self.assertEqual(build_mineru_env("cuda:1")["MINERU_DEVICE_MODE"], "cuda:1")

    def test_prepare_and_render_mock_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "paper.pdf"
            doc = fitz.open()
            page = doc.new_page(width=300, height=400)
            page.insert_text((30, 50), "Introduction", fontsize=14)
            page.insert_text((30, 90), "The attention score is important.", fontsize=10)
            page.insert_text((30, 140), "References", fontsize=14)
            page.insert_text((30, 180), "Vaswani et al. 2017.", fontsize=10)
            doc.save(pdf_path)
            doc.close()

            mineru_dir = root / "mineru"
            mineru_dir.mkdir()
            content = [
                {"type": "text", "text": "Introduction", "text_level": 1, "bbox": [100, 100, 700, 150], "page_idx": 0},
                {
                    "type": "text",
                    "text": "The attention score is important.",
                    "bbox": [100, 200, 900, 240],
                    "page_idx": 0,
                },
                {"type": "text", "text": "References", "text_level": 1, "bbox": [100, 350, 700, 400], "page_idx": 0},
                {"type": "text", "text": "Vaswani et al. 2017.", "bbox": [100, 450, 900, 490], "page_idx": 0},
            ]
            (mineru_dir / "paper_content_list.json").write_text(json.dumps(content), encoding="utf-8")

            ir = build_document_ir(pdf_path, mineru_dir, protected_terms=["attention"])
            payload = build_payload(ir)
            # b_00000 is the "Introduction" heading, kept in the source language.
            self.assertEqual([b["id"] for b in payload["blocks"]], ["b_00001"])
            self.assertTrue(ir.blocks[2].excluded)
            self.assertTrue(ir.blocks[3].excluded)

            response = build_mock_response(payload, prefix="번역: ")
            output_pdf = root / "rendered.pdf"
            warnings = render_pdf(pdf_path, ir, response, output_pdf)
            self.assertTrue(output_pdf.exists())
            self.assertIsInstance(to_jsonable(ir), dict)
            self.assertTrue(all("missing response" not in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
