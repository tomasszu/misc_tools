import torch
import torch.nn as nn
import torch.onnx

from vehicle_reid.load_model import load_model_from_opts

# Switch model to CPU for export
device = "cpu"
model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/main_sp4_ep6_ft_noCEL_070126/opts.yaml", 
                                ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/main_sp4_ep6_ft_noCEL_070126/net_26.pth", 
                                remove_classifier=True)
model.to(device)
model.eval()

# Remove classifier block if it exists
#model.classifier.add_block[2] = nn.Sequential()  # Šķietami unnnescessary, tomēr ar .eval() vajadzētu pietikt, lai noņemtu dropout un batchnorm


# model.eval()
# for module in model.modules():
#     module.eval()

# Dummy input in expected input shape (e.g., 3×256×128)
dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

with torch.no_grad():
    assert model(dummy_input).shape.numel() == 256, "Last linear layer assertion failed" # Checking that the classifier head has really been removed

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/main_sp4_ep6_ft_noCEL_070126/sp4_ep6_ft_noCEL_070126_26ep.onnx",
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    do_constant_folding=True
)
