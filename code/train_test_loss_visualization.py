
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Drawn at the printed width (0.6\linewidth = 3.3in); NeurIPS does not accept Type 3 fonts
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'axes.titleweight': 'bold',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
output_folder = r"C:\Temp\Unlearning\MIA Critique Paper\NeuroIPS Submission\Camera Ready\figures"
os.makedirs(output_folder, exist_ok=True)

# === Log paths and subplot titles ===
log_files = {
    r"C:\Temp\Unlearning\Base Models\allcnn_cifar10.txt": "AllCNN on CIFAR-10",
    r"C:\Temp\Unlearning\Base Models\resnet_cifar10.txt": "ResNet-18 on CIFAR-10",
    r"C:\Temp\Unlearning\Base Models\vit_cifar10.txt": "ViT on CIFAR-10",
    r"C:\Temp\Unlearning\Base Models\allcnn_cifar20.txt": "AllCNN on CIFAR-20",
    r"C:\Temp\Unlearning\Base Models\resnet_cifar20.txt": "ResNet-18 on CIFAR-20",
    r"C:\Temp\Unlearning\Base Models\allcnn_mnist.txt": "AllCNN on MNIST",
    r"C:\Temp\Unlearning\Base Models\resnet_mnist.txt": "ResNet-18 on MNIST",
    r"C:\Temp\Unlearning\Base Models\resnet_mucac.txt": "ResNet-18 on MUCAC",
    
    r"C:\Temp\Unlearning\Base Models\ResNet18-Cifar10-Overfitted-best.txt": "ResNet-18 on Cifar-10 Overfitted"
    # Add more as needed
}
# === Regex patterns ===
train_pattern = re.compile(r"Epoch (\d+) - Average Train Loss: ([\d.]+), Train Accuracy: ([\d.]+)")
test_pattern = re.compile(r"Test set: Epoch: (\d+), Average loss: ([\d.]+), Accuracy: ([\d.]+)")

# === Process each log ===
for log_path, title in log_files.items():
    train_acc = {}
    test_acc = {}

    with open(log_path, 'r') as f:
        for line in f:
            if match := train_pattern.search(line):
                epoch = int(match.group(1))
                train_acc[epoch] = float(match.group(3))
            elif match := test_pattern.search(line):
                epoch = int(match.group(1))
                test_acc[epoch] = float(match.group(3))

    # Only plot if both sets exist
    epochs = sorted(set(train_acc) & set(test_acc))
    if not epochs:
        continue

    # Prepare accuracy values
    train_acc_vals = [train_acc[e] for e in epochs]
    test_acc_vals = [test_acc[e] for e in epochs]

    # === Plot ===
    plt.figure(figsize=(3.3, 1.9))
    plt.plot(epochs, train_acc_vals, label='Train Accuracy', color='green', linewidth=1.0)
    plt.plot(epochs, test_acc_vals, label='Test Accuracy', color='red', linewidth=1.0)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="best", frameon=True)
    plt.grid(True)
    plt.tight_layout(pad=0.3)

    # Save plot
    filename = "figure_" + title.lower().replace("(", "").replace(")", "").replace(" ", "_").replace("-", "") + ".pdf"
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")
