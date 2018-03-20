import torch
import utils


class PadRemover(object):
    """Helper to remove padding from a tensor before sending to the experts.

  The padding is computed for one reference tensor containing the padding mask
  and then can be applied to any other tensor of shape [dim_origin,...].

  Ex:
      input = [
        [tok1, tok2],
        [tok3, tok4],
        [0, 0],
        [0, 0],
        [tok5, tok6],
        [0, 0],
      ]
      output = [
        [tok1, tok2],
        [tok3, tok4],
        [tok5, tok6],
      ]
  """

    def __init__(self, pad_mask):
        """Compute and store the location of the padding.

      Args:
      pad_mask (Tensor): Reference padding tensor of shape
      [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
      containing non-zeros positive values to indicate padding location.
      """
        pad_mask = pad_mask.view(-1)
        self.nonpad_ids = pad_mask.nonzero()[:, 0]
        self.dim_origin = pad_mask.size()[0]

    def remove(self, x):
        """Remove padding from the given tensor.
    Args:
      x (Tensor): of shape [dim_origin,...]

    Returns:
      a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
    """
        x_shape = x.size()
        x = torch.index_select(x, dim=0, index=self.nonpad_ids)
        return x

    def restore(self, x):
        """Add padding back to the given tensor.
    Args:
      x (Tensor): of shape [dim_compressed,...]

    Returns:
      a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
      dim is restored from the original reference tensor
    """
        z = torch.autograd.Variable(torch.zeros(self.dim_origin, x.size()[1]).type(utils.FLOAT_TYPE),
                                    requires_grad=False)
        z.index_copy_(0, self.nonpad_ids, x)
        return z
