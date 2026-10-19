import json
import torch
import torch.nn as nn
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.training.train_utils import evaluate_extended
from src.data.backdoor import create_asr_dataloader
from torch.utils.data import DataLoader, Subset

device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

batch_size = 64
_, test_loader_full = get_cifar10_dataloaders(batch_size=batch_size, use_fake=False)
test_dataset = test_loader_full.dataset

eval_clean_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

non_airplane_indices = [i for i, label in enumerate(test_dataset.targets) if label != 0]
eval_non_airplane_dataset = Subset(test_dataset, non_airplane_indices)

eval_asr_loader = create_asr_dataloader(test_loader_full, batch_size=batch_size, target_class=0, trigger_size=3)
eval_untriggered_loader = DataLoader(eval_non_airplane_dataset, batch_size=batch_size, shuffle=False)

airplane_indices = [i for i, label in enumerate(test_dataset.targets) if label == 0]
eval_airplane_dataset = Subset(test_dataset, airplane_indices)
eval_airplane_loader = DataLoader(eval_airplane_dataset, batch_size=batch_size, shuffle=False)

global_model = SmallCNN().to(device)
global_model.load_state_dict(torch.load("results/iid_clean/final_model.pt", map_location=device, weights_only=True))
global_model.eval()

criterion = nn.CrossEntropyLoss()

clean_metrics = evaluate_extended(global_model, eval_clean_loader, criterion, device)
asr_metrics = evaluate_extended(global_model, eval_asr_loader, criterion, device)

utr_correct = 0
utr_total = 0
with torch.no_grad():
    for x, _ in eval_untriggered_loader:
        x = x.to(device)
        p = global_model(x).argmax(1)
        utr_correct += (p == 0).sum().item()
        utr_total += x.size(0)
utr = (utr_correct / utr_total) * 100.0 if utr_total > 0 else 0.0

_, airplane_recall = evaluate_extended(global_model, eval_airplane_loader, criterion, device).get("accuracy", 0), evaluate_extended(global_model, eval_airplane_loader, criterion, device)["accuracy"]

# Write to a patch file
out = {
    "clean_accuracy": clean_metrics["accuracy"],
    "clean_loss": clean_metrics["loss"],
    "clean_macro_f1": clean_metrics["macro_f1"] * 100.0,
    "triggered_rate": asr_metrics["accuracy"],
    "untriggered_airplane_rate": utr,
    "airplane_recall": airplane_recall
}
with open("results/iid_clean/extended_metrics.json", "w") as f:
    json.dump(out, f, indent=4)
print("Saved extended_metrics.json for iid_clean!")
