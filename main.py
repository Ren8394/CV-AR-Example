from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from datasets import HAR
from models import CNN, VGG16

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using {DEVICE}")

CLASS_LIST = [
    "sitting", "using_laptop", "hugging", "sleeping", "drinking", 
    "clapping", "dancing", "cycling", "calling", "laughing",
    "eating", "fighting", "listening_to_music", "running", "texting"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Human Activity Recognition")
    parser.add_argument("--model", type=str, default="vgg", help="model name")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # model
    if args.model == "cnn":
        model = CNN(num_classes=15).to(DEVICE)
    elif args.model == "vgg":
        model = VGG16(num_classes=15).to(DEVICE)

    # hyperparameters
    batch_size = 32
    lr = 1e-4
    epochs = 25

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # dataset
    train_dataloader = DataLoader(HAR("train"), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(HAR("val"), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(HAR("test"), batch_size=batch_size, shuffle=False)

    # train and validation
    Path("results").mkdir(exist_ok=True)
    Path("weights").mkdir(exist_ok=True)
    best_loss = np.inf
    train_loss_list = []
    vals_loss_list = []
    for epoch in trange(epochs):
        # training
        model.train()
        train_loss = 0.0
        for _, (x, y) in tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader), desc=f"epoch: {epoch+1}/{epochs}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_loss_list.append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        for _, (x, y) in tqdm(enumerate(val_dataloader), leave=False, total=len(val_dataloader), desc=f"Validation: "):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.no_grad():
                y_hat = model(x)
                loss = criterion(y_hat, y)
            val_loss += loss.item()
        val_loss /= len(val_dataloader)
        vals_loss_list.append(val_loss)

        # save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"weights/best_{args.model}.pth")

        # print loss
        print(f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    np.savetxt(f"results/train_loss_{args.model}.txt", train_loss_list, fmt="%.4f")
    np.savetxt(f"results/val_loss_{args.model}.txt", vals_loss_list, fmt="%.4f")

    # test
    model.load_state_dict(torch.load(f"weights/best_{args.model}.pth", map_location=DEVICE))
    model.eval()
    pred_class_list = []
    true_class_list = []
    for _, (x, y) in tqdm(enumerate(test_dataloader), leave=False, total=len(test_dataloader), desc=f"Test: "):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            y_hat = model(x)
            pred_classes=torch.softmax(y_hat, dim=1).argmax(dim=1).cpu().numpy()
            # pred_classes=y_hat.argmax(dim=1).cpu().numpy()
            pred_class_list.extend(pred_classes)
            true_classes=y.argmax(dim=1).cpu().numpy()
            true_class_list.extend(true_classes)
    
    np.savetxt(f"results/pred_class_count_{args.model}.txt", pred_class_list, fmt="%d")
    np.savetxt(f"results/true_class_count_{args.model}.txt", true_class_list, fmt="%d")
