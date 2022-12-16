import config
from model import YOLOv3
from utils import load_checkpoint, check_class_accuracy, get_loaders
import torch.optim as optim
import torch

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    optimizer = optim.Adam( model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")

    check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
