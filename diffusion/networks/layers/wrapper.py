import torch
import torch.nn as nn

# Max B*E allowed in a single batched forward through wrapped PyG conv.
# Beyond ~80k, gatv2_conv.add_self_loops triggers a scatter-related crash with
# v2.61 dataset (max_instance=1600, dense edges). Chunking the batch dim
# bypasses the issue; B*E threshold below works for both v1.61-fs and v2.61.
_BATCH_WRAPPER_MAX_BE = 80_000


class BatchWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, edge_index, edge_attr=None, **kwargs):
        # process x, edge_index, edge_attr
        # x: (B, V, F)
        # edge_index: (2, E)
        # edge attributes: (E, F)
        B, V, F = x.shape
        _, E = edge_index.shape

        # Auto-chunk to avoid the PyG scatter limit on large B*E (v2.61 dataset).
        # For v1.61-fs (E~1500), B<=32 keeps B*E<48k → single forward.
        # For v2.61 (E~18k+), B>4 exceeds the limit → chunked forward.
        max_chunk = max(1, _BATCH_WRAPPER_MAX_BE // max(E, 1))
        if B <= max_chunk:
            return self._forward_chunk(x, edge_index, edge_attr, **kwargs)

        outputs = []
        for b_start in range(0, B, max_chunk):
            b_end = min(b_start + max_chunk, B)
            out = self._forward_chunk(x[b_start:b_end], edge_index, edge_attr, **kwargs)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    def _forward_chunk(self, x, edge_index, edge_attr=None, **kwargs):
        B, V, F = x.shape
        _, E = edge_index.shape

        x_unbatched = x.reshape(B * V, F)

        if edge_attr is not None:
            edge_attr_unbatched = edge_attr.view(1, *edge_attr.shape).expand(B, -1, -1)
            edge_attr_unbatched = edge_attr_unbatched.reshape(B * E, -1)
        else:
            edge_attr_unbatched = None

        edge_index_unbatched = edge_index.movedim(-1, 0).unsqueeze(dim=0).expand(B, E, 2)
        edge_index_offset = torch.arange(0, V*B, V, device=edge_index.device, dtype=edge_index.dtype).view(B, 1, 1)
        edge_index_unbatched = edge_index_unbatched + edge_index_offset
        edge_index_unbatched = edge_index_unbatched.reshape(B * E, 2).movedim(0, -1)

        output_unbatched = self.net(x_unbatched, edge_index_unbatched, edge_attr=edge_attr_unbatched, **kwargs)
        output = output_unbatched.view(B, V, -1)
        return output