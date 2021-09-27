import time
from tqdm.auto import tqdm
import torch
from search_utils import step_LR, adam_optim, sgd_optim, bce_loss, bce_logit_loss, crossentropy_loss
from preprocessing import prep_dataloaders


def make_optims(model, criterion, optimizer, scheduler, lr):
    criterion_dic = {"bce_loss": bce_loss,
                     "bce_logit_loss": bce_logit_loss,
                     "crossentropy_loss": crossentropy_loss}
    criterion = criterion_dic[criterion]()

    optimizer_dic = {"sgd": sgd_optim,
                     "adam": adam_optim}
    optimizer = optimizer_dic[optimizer](model.parameters(), lr)

    scheduler = step_LR(optimizer)
    return (criterion, optimizer, scheduler)


def train_model(model_class, dataloader_dict, criterion, optimizer, scheduler, learning_rate, device, batch_size, num_epochs=10):
    since = time.time()
    model = model_class.model

    model_class.update_best_state_dict()
    best_acc = 0.0

    criterion, optimizer, scheduler = make_optims(model, criterion, optimizer, scheduler, learning_rate)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for _, (inputs, labels) in tqdm(enumerate(dataloader_dict[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (batch_size * len(dataloader_dict[phase]))
            epoch_acc = running_corrects.double() / (batch_size * len(dataloader_dict[phase]))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_class.update_best_state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(f'Done with model {model_class.name}')
    return best_acc


def run_search(models: list, train, criterion: str="crossentropy_loss", optimizer: str="adam", scheduler: str="step", num_epochs: int=10, device='cpu', val_split: float=0.2, batch_size: int=16, shuffle: bool=True, learning_rate: float=0.001):
    best_model = (None, 0)
    device = torch.device(device)
    print(f"Using {device}")
    dataloader_dict = prep_dataloaders(train, val_split, batch_size, shuffle)
    for model in models:
        print(f"Running with model {model.name} using {num_epochs} epochs")
        acc = train_model(model, dataloader_dict, criterion, optimizer, scheduler, learning_rate, device, batch_size, num_epochs)
        if acc > best_model[1]:
            best_model = (model, acc)
    print("Done!")
    