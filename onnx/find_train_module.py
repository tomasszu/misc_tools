import torch

from vehicle_reid.load_model import load_model_from_opts

# Switch model to CPU for export
device = "cpu"
model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", 
                                ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", 
                                remove_classifier=True)
model.to(device)
model.eval()

for name, module in model.named_modules():
    if hasattr(module, 'training') and module.training:
        print(f"Module {name} is still in training mode.")
    else:
        print(f"Module {name} is in evaluation mode.")
