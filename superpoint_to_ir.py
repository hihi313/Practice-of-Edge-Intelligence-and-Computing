 # Load the network in inference mode

from models.superpoint.models import SuperPointNet_gauss2

net = SuperPointNet_gauss2()
checkpoint = torch.load(weights_path)
net.load_state_dict(checkpoint["model_state_dict"])
net.to(device)
net.eval()