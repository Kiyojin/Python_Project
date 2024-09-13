import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from architecture import MyCNN
from dataloader import get_dataset, stacking


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 3,
        show_progress: bool = False,
) -> tuple[list, list]:
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using {device}")

    # Vars for early stopping
    best_eval_loss = float('inf')
    patience_counter = 0

    # Vars for training
    network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_function = torch.nn.CrossEntropyLoss()

    # Data loaders
    training_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=stacking)
    eval_loader = DataLoader(
        eval_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=stacking)

    train_losses = []
    eval_losses = []

    for epoch in tqdm(range(num_epochs)) if show_progress else range(num_epochs):

        # training loop
        network.train()
        total_train_loss = 0
        for train_data in training_loader:
            # Reset gradients
            optimizer.zero_grad()

            # Load minibatch features
            images, _, _, classids, classnames, _ = train_data
            if images.device != device:  # Check if images are already on the device
                images = images.to(device=device)
            if classids.device != device:  # Check if classids are already on the device
                classids = classids.to(device=device)

            # Compute the output
            output = network(images)
            # Convert classids to 1D tensor
            classids_1d = torch.squeeze(classids.long())  # Can use torch.flatten() as well

            # Compute loss
            train_loss = loss_function(output, classids_1d)

            # Add L2 regularization
            l2_term = torch.mean(torch.stack([(param ** 2).mean() for param in network.parameters()]))
            # Compute final loss
            train_loss = train_loss + l2_term * 1e-2

            # Accumulate batch losses
            total_train_loss += train_loss.item()

            # Compute the gradients
            train_loss.backward()
            # Preform the update
            optimizer.step()

        # Average Train loss
        train_loss_avg = total_train_loss / len(training_loader)
        train_losses.append(train_loss_avg)

        # evaluation loop
        network.eval()
        total_eval_loss = 0
        correct_predictions = 0  # counter for correct predictions
        total_predictions = 0  # counter for total predictions
        for eval_data in eval_loader:
            # Load minibatch features
            images, _, _, classids, classnames, _ = eval_data
            if images.device != device:  # Check if images are already on the device
                images = images.to(device=device)
            if classids.device != device:  # Check if classids are already on the device
                classids = classids.to(device=device)

            # No gradient computations needed
            with torch.no_grad():
                output = network(images)
                # Convert classids to 1D tensor
                classids_1d = torch.squeeze(classids.long())  # Can use torch.flatten() as well
                # Compute loss
                eval_loss = loss_function(output, classids_1d)
                # Accumulate batch losses
                total_eval_loss += eval_loss.item()

                # Prediction
                _, predicted = torch.max(output.data, 1)
                total_predictions += classids_1d.size(0)
                correct_predictions += (predicted == classids_1d).sum().item()

        # Average Eval loss
        eval_loss_avg = total_eval_loss / len(eval_loader)
        eval_losses.append(eval_loss_avg)

        # Eval accuracy
        eval_accuracy = correct_predictions / total_predictions

        scheduler.step()

        # Update after every Epoch
        print(f"\nAverage Training Loss for Epoch {epoch}: {train_loss_avg}"
              f"\nAverage Evaluation Loss for Epoch {epoch}: {eval_loss_avg}"
              f"\nEvaluation Accuracy for Epoch {epoch}: {eval_accuracy}")

        # Early stopping
        if eval_loss_avg < best_eval_loss:
            # Set new best eval_loss
            best_eval_loss = eval_loss_avg
            patience_counter = 0
            # Save Model
            torch.save(network.state_dict(), "model.pth")
            print(f"\nModel saved")
        else:
            # No improvement -> increase patience counter
            patience_counter += 1

            # if the patience counter reaches threshold (patience), stop training
            if patience_counter >= patience:
                print("Early stopping, no improvement for " + str(patience) + " epochs...")
                break


    return train_losses, eval_losses



def plot_losses(train_losses: list, eval_losses: list):
    # Make a figure
    fig = plt.figure()

    # Create an axes
    ax = plt.axes()

    # Move to cpu, detach from computational graph and convert to numpy
    train_losses_cpu = [tensor.detach().cpu().numpy() for tensor in train_losses]
    eval_losses_cpu = [tensor.detach().cpu().numpy() for tensor in eval_losses]

    # plot data
    ax.plot(range(len(train_losses_cpu)), train_losses_cpu, label='Train Loss', color='blue')
    ax.plot(range(len(eval_losses_cpu)), eval_losses_cpu, label='Eval Loss', color='green')

    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')

    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    # Setup labels, title, and legend
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean-Squared Error Loss")
    ax.legend()
    ax.set_title("Loss Function")

    # Save the figure as a pdf
    plt.savefig("epoch_loss.pdf")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    train_data, eval_data = get_dataset()
    model = MyCNN(
        input_channels=1,
        hidden_channels=[32, 64, 128],
        image_size=(100, 100),
        use_batchnormalization=True,
        kernel_size=[3, 5, 7],
        num_classes=20,
        activation_function=torch.nn.ELU())
    train_losses, eval_losses = training_loop(model, train_data, eval_data, show_progress=True)
    plot_losses(train_losses, eval_losses)
