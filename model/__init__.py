from sys import version_info
version = version_info.major
if version==2:
  from base_model import Model
  from simple_cnn_model import SimpleCNNLocalEntLinkModel
  from simple_ctx_cnn_model import SimpleCtxCNNLocalEntLinkModel
  from rdgcn_global_model import RDGraphCNNGlobalEntLinkModel
else:
  from .base_model import Model
  from .simple_cnn_model import SimpleCNNLocalEntLinkModel
  from .simple_ctx_cnn_model import SimpleCtxCNNLocalEntLinkModel
  from .rdgcngcn_global_model import RDGraphCNNGlobalEntLinkModel