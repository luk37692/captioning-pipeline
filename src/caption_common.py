"""Beam search shared by the captioner. The concrete captioner subclasses
`BeamCaptioner` and only provides its specifics (feature extraction + one decode step)."""
import numpy as np

import config

_SPECIAL = ("<start>", "<end>", "<pad>")


class BeamCaptioner:
    """Beam search + repeated n-gram blocking.

    A weakly trained decoder tends to loop ("... person up of the ..."), hence the
    n-gram blocking. Subclasses must define, after init:
      - word_index / index_word
      - start_id / end_id (end_id = -1 if absent from vocab)
      - unk_id (None = no <unk> blocking)
      - n_steps (max number of decode steps)
    and implement `_features(img01)`, `_init_state()`, `_decode_step(seq, features, state)`.
    """
    unk_id = None

    @staticmethod
    def _banned_tokens(seq, n):
        """Tokens forbidden at the next step to avoid repeating an already-seen n-gram."""
        if n <= 0 or len(seq) < n:
            return set()
        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
        banned = set()
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
        return banned

    def _init_state(self):
        return None

    def _decode_step(self, seq, features, state):
        """Returns (numpy log_probs of the next token, new state)."""
        raise NotImplementedError

    def caption(self, img01, beam_width=None, no_repeat_ngram_size=3):
        """Caption an HxWx3 float [0,1] image (beam search + n-gram blocking)."""
        beam_width = beam_width or config.MAX_CAPTION_BEAM
        features = self._features(img01)
        beams = [(0.0, [self.start_id], self._init_state(), False)]
        for _ in range(self.n_steps):
            candidates = []
            for logp, seq, state, done in beams:
                if done:
                    candidates.append((logp, seq, state, True))
                    continue
                lp, new_state = self._decode_step(seq, features, state)
                if self.unk_id is not None:
                    lp[self.unk_id] = -np.inf
                banned = self._banned_tokens(seq, no_repeat_ngram_size)
                taken = 0
                for t in np.argsort(lp)[::-1]:
                    t = int(t)
                    if t in banned and t != self.end_id:
                        continue
                    candidates.append((logp + float(lp[t]), seq + [t], new_state, t == self.end_id))
                    taken += 1
                    if taken >= beam_width:
                        break
            candidates.sort(key=lambda b: b[0] / (len(b[1]) ** 0.7), reverse=True)
            beams = candidates[:beam_width]
            if all(b[3] for b in beams):
                break
        best = max(beams, key=lambda b: b[0] / (len(b[1]) ** 0.7))
        words = [self.index_word.get(i, "<unk>") for i in best[1]]
        return " ".join(w for w in words if w not in _SPECIAL)
