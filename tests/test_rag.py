import unittest

from agenthub.rag.bm25 import BM25
from agenthub.rag.chunker import split_markdown
from agenthub.rag.retriever import HybridRetriever
from agenthub.textutil import tokenize
from tests.helpers import KB


class RagTests(unittest.TestCase):
    def test_tokenize_mixed(self):
        t = tokenize("P1工单 SLA")
        self.assertIn("p1", t); self.assertIn("sla", t); self.assertIn("工单", t)

    def test_chunker_keeps_heading_path_and_splits_long(self):
        md = "# 文档\n## 小节A\n" + "内容。" * 400 + "\n## 小节B\n短内容"
        chunks = split_markdown(md, "d.md", max_chars=200, overlap=20)
        self.assertTrue(all(c.heading in ("小节A", "小节B") for c in chunks))
        self.assertGreater(len([c for c in chunks if c.heading == "小节A"]), 1)
        self.assertTrue(all(len(c.text) <= 260 for c in chunks))
        self.assertEqual(chunks[0].source, "d.md > 小节A")

    def test_bm25_prefers_matching_doc(self):
        s = BM25(["退货运费由买家承担", "积分有效期24个月"]).scores("积分多久过期")
        self.assertGreater(s[1], s[0])

    def test_hybrid_retrieval_expected_sources(self):
        r = HybridRetriever.from_dir(KB)
        cases = {"定制类商品能退货吗": "退换货政策.md", "满多少钱包邮": "物流与配送.md",
                 "P1工单多久必须响应": "内部IT工单SLA.md", "密码多久更换": "信息安全规范.md"}
        for q, doc in cases.items():
            self.assertEqual(r.search(q, 1)[0].chunk.doc, doc, q)

    def test_doc_filter_and_empty_query(self):
        r = HybridRetriever.from_dir(KB)
        self.assertTrue(all(h.chunk.doc == "物流与配送.md" for h in r.search("退货", 5, doc="物流与配送.md")))
        self.assertEqual(r.search("zzzzqqq", 3), [])


if __name__ == "__main__":
    unittest.main()
