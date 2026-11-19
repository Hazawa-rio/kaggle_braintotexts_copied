"""
集束搜尋
========

對應原 notebook：Cell 2 的 BeamSearchGenerator
拆解筆記：4.11 ①

每一步保留分數最高的 beam_width 條路線，避免「一步錯、步步錯」。
路線分數 = 每個字母的對數機率加總。

問題 P10：max_len 預設 80，長句會被截斷
問題 P11：沒有長度正規化，越長的句子分數越低，偏好短句
注意：這裡用的是正確的 bos_id = 1、eos_id = 2，沒有 P3 的問題
"""

import torch


class BeamSearchGenerator:

    def __init__(self, model, tokenizer, beam_width=5, max_len=80):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_len = max_len

    @torch.no_grad()
    def generate(self, z_neural):
        """回傳分數最高的一條路線（token 編號）。原作者有寫但最終流程沒用到。"""
        device = z_neural.device
        sequences = [[self.tokenizer.bos_id]]
        scores = [0.0]

        for step in range(self.max_len):
            all_candidates = []
            for seq, score in zip(sequences, scores):
                if seq[-1] == self.tokenizer.eos_id:
                    all_candidates.append((seq, score))
                    continue
                seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                logits = self.model.text_decoder(z_neural, seq_tensor)
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, self.beam_width)
                for prob, idx in zip(topk_probs, topk_ids):
                    all_candidates.append((seq + [idx.item()], score + prob.item()))

            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [seq for seq, _ in all_candidates[:self.beam_width]]
            scores = [score for _, score in all_candidates[:self.beam_width]]
            if all(seq[-1] == self.tokenizer.eos_id for seq in sequences):
                break

        return torch.tensor(sequences[0])

    @torch.no_grad()
    def generate_candidates(self, z_neural):
        """
        回傳最後留下的所有候選句（文字），最終流程用的是這個。

        輸入：z_neural (1, L, 512)，一次只處理一句
        輸出：beam_width 個字串，依分數由高到低
        """
        device = z_neural.device
        sequences = [[self.tokenizer.bos_id]]
        scores = [0.0]

        for step in range(self.max_len):
            all_candidates = []
            all_beams_ended = True

            for seq, score in zip(sequences, scores):
                if seq[-1] == self.tokenizer.eos_id:
                    all_candidates.append((seq, score))
                    continue
                all_beams_ended = False

                # 每條路線每一步都重新跑一次完整的解碼器（沒有快取，所以很慢）
                seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                logits = self.model.text_decoder(z_neural, seq_tensor)
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, self.beam_width)
                for prob, idx in zip(topk_probs, topk_ids):
                    all_candidates.append((seq + [idx.item()], score + prob.item()))

            if all_beams_ended:
                break

            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [seq for seq, _ in all_candidates[:self.beam_width]]
            scores = [score for _, score in all_candidates[:self.beam_width]]

        return [self.tokenizer.decode(seq) for seq in sequences]
