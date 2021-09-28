import time
import pandas as pd
from tqdm.auto import tqdm
import torch
from AutoDL.search_utils import step_LR, adam_optim, sgd_optim, bce_loss, bce_logit_loss, crossentropy_loss
from AutoDL.preprocessing import prep_dataloaders, put_data_to_dataloader
from torch.nn import DataParallel


class Search():
    def __init__(self, models: list, train, device='cpu', val_split: float=0.2, batch_size: int=16, shuffle: bool=True, learning_rate: float=0.001):
        self.best_model = None
        self.best_model_score = 0
        self.results = {"Model": [], "Model Name": [], "Validation Accuracy": []}
        self.models = models
        self.train = train
        self.device = device
        self.val_split = val_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate


    def get_rankings(self):
        return pd.DataFrame().from_dict(self.results)


    def make_optims(self, model, criterion, optimizer, scheduler, lr):
        criterion_dic = {"bce_loss": bce_loss,
                        "bce_logit_loss": bce_logit_loss,
                        "crossentropy_loss": crossentropy_loss}
        criterion = criterion_dic[criterion]()

        optimizer_dic = {"sgd": sgd_optim,
                        "adam": adam_optim}
        optimizer = optimizer_dic[optimizer](model.parameters(), lr)

        scheduler = step_LR(optimizer)
        return (criterion, optimizer, scheduler)


    def train_model(self, model_class, dataloader_dict, criterion, optimizer, scheduler, learning_rate, batch_size, num_epochs=10):
        model = model_class.model
        best_acc = 0.0

        model = DataParallel(model)
        criterion, optimizer, scheduler = self.make_optims(model, criterion, optimizer, scheduler, learning_rate)
        since = time.time()
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
                for _, (inputs, labels) in tqdm(enumerate(dataloader_dict[phase]), total=len(dataloader_dict[phase])):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

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

        time_elapsed = time.time() - since
        print('{}: Training complete in {:.0f}m {:.0f}s'.format(
            model_class.name, time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc), "\n")
        return best_acc


    def test_model(self, model_class, data):
        dataloader = put_data_to_dataloader(data, batch_size=self.batch_size)
        model = model_class.model
        correct = 0
        total = 0

        model = DataParallel(model)
        with torch.no_grad():
            for data in tqdm(dataloader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} test images: {}%'.format(total, 100 * correct / total))


    def run_search(self, criterion: str="crossentropy_loss", optimizer: str="adam", scheduler: str="step", num_epochs: int=10):
        best_model = (None, 0)
        self.device = torch.device(self.device)
        print(f"Using {self.device}")
        dataloader_dict = prep_dataloaders(self.train, self.val_split, self.batch_size, self.shuffle)

        for model in self.models:
            print(f"Running with model {model.name} using {num_epochs} epochs")
            acc = self.train_model(model, dataloader_dict, criterion, optimizer, scheduler, self.learning_rate, self.batch_size, num_epochs)

            if acc > self.best_model_score:
                self.best_model = model
                self.best_model_score = acc
            self.results["Validation Accuracy"].append(acc.item())
            self.results["Model Name"].append(model.name)
            self.results["Model"].append(model)

        print("Done!")
