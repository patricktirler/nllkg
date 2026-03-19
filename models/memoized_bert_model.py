from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F
from mmdet.models.language_models.bert import BertModel
from mmdet.registry import MODELS


@MODELS.register_module()
class MemoizedBertModel(BertModel):
    """BertModel that caches results per unique caption string.

    Each caption is always run through BERT individually (batch size 1)
    so cached entries are fully deterministic — independent of whatever
    other captions happen to be in the same batch.

    On a forward call the batch result is assembled by padding each cached
    entry to the same token length and stacking along the batch dimension.

    Args:
        use_cache: Set False to disable caching (pass-through).
    """

    def __init__(self, use_cache: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = use_cache
        self._cache: Dict[str, dict] = {}

    def forward(self, captions: Sequence[str], **kwargs) -> Dict[str, Any]:
        if not self.use_cache:
            return super().forward(captions, **kwargs)

        with torch.no_grad():
            # Run BERT individually for any caption not yet cached.
            # Single-string forward ensures each entry is computed identically
            # regardless of what other captions are in the batch.
            for caption in captions:
                if caption not in self._cache:
                    result = super().forward([caption], **kwargs)
                    self._cache[caption] = {
                        k: v.detach().clone() for k, v in result.items()
                    }

            # Assemble the batch: pad each (1, T_i, ...) entry to the same
            # token length, then stack into (B, T_max, ...).
            entries = [self._cache[c] for c in captions]
            return {
                k: self._pad_and_stack([e[k] for e in entries])
                for k in entries[0]
            }

    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def _pad_and_stack(tensors: list) -> torch.Tensor:
        """Pad tensors to the same token length (dim 1) and stack.

        Handles all text_dict shapes:
          - (1, T)       position_ids, text_token_mask  -> pad dim 1
          - (1, T, T)    masks (block-diagonal)         -> pad both token dims
          - (1, T, D)    embedded, hidden               -> pad dim 1, leave D
        """
        max_len = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            pad = max_len - t.shape[1]
            if pad == 0:
                padded.append(t)
            elif t.dim() == 2:
                fill = False if t.dtype == torch.bool else 0
                padded.append(F.pad(t, (0, pad), value=fill))
            elif t.dim() == 3 and t.shape[1] == t.shape[2]:  # (1, T, T)
                padded.append(F.pad(t, (0, pad, 0, pad), value=0))
            else:                                              # (1, T, D)
                padded.append(F.pad(t, (0, 0, 0, pad), value=0))
        return torch.cat(padded, dim=0)