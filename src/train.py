import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

import src.config as config_file
import src.utils.metrics as metrics
import src.model.rythmnet as rythmnet
import src.utils.loss_func as loss_func
import src.dataset.data_loader as data_loader
import src.dataset.signal_transforms as signal_transforms


def train_fn(model, optimizer, train_dataloader, loss_fn, params):
    running_loss = 0.0
    target_hr_list = []
    predicted_hr_list = []

    for batch_idx, batch_data in enumerate(train_dataloader):
        print('batch_idx', batch_idx)
        # Limiting training data for faster epochs.
        if batch_idx * params['BATCH_SIZE'] >= params['N_TRAIN_EXAMPLES']:
            break
        # batch = next(iter(train_dataloader))
        st_maps_batch = batch_data['st_maps']
        labels_batch = batch_data['target']

        for batch_n in range(labels_batch.shape[0]):
            # Every data instance is an input + label pair
            inputs = st_maps_batch[batch_n, :, :, :, :]
            labels = labels_batch[batch_n, :]

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Make predictions for this batch
                reg_outputs, gru_outputs = model(inputs)

                # Compute the loss
                loss = loss_fn(reg_outputs.squeeze(0), gru_outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

            # Gather data and report
            target_hr_list.append(labels.mean().item())
            predicted_hr_list.append(gru_outputs.mean().item())

            running_loss += loss.item()
    last_loss = running_loss / ((batch_idx+1) * params['BATCH_SIZE'])

    return target_hr_list, predicted_hr_list, last_loss


def val_fn(model, val_dataloader, loss_fn, params):
    val_loss = 0.0
    target_hr_list = []
    predicted_hr_list = []

    for batch_idx, batch_data in enumerate(val_dataloader):
        print('batch_idx', batch_idx)
        # Limiting training data for faster epochs.
        if batch_idx * params['BATCH_SIZE'] >= params['N_VAL_EXAMPLES']:
            break

        st_maps_batch = batch_data['st_maps']
        labels_batch = batch_data['target']

        with torch.no_grad():
            for batch_n in range(labels_batch.shape[0]):
                # Every data instance is an input + label pair
                inputs = st_maps_batch[batch_n, :, :, :, :]
                labels = labels_batch[batch_n, :]

                # Make predictions for this batch
                reg_outputs, gru_outputs = model(inputs)
                loss = loss_fn(reg_outputs.squeeze(0), gru_outputs, labels)
                val_loss += loss.item()

                target_hr_list.append(labels.mean().item())
                predicted_hr_list.append(gru_outputs.mean().item())

    last_fin_loss = val_loss / ((batch_idx+1) * params['BATCH_SIZE'])
    # print('batch {} loss: {}'.format(batch_idx + 1, last_fin_loss))

    return target_hr_list, predicted_hr_list, last_fin_loss


def save_model_checkpoint(model, optimizer, train_loss, checkpoint_path, save_filename = "running_model.pt"):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, os.path.join(checkpoint_path, save_filename))
    print('Saved!')


def load_dataset(params):

    df_experiment = data_loader.build_df_experiment_leave_out(
        list_train_participants=params['PARTICIPANTS_TRAIN_VAL'],
        ratio_train_val=params['RATIO_DATA_TRAIN_VAL'],
        leave_out_test_participants=params['PARTICIPANTS_TEST'],
        dataset_path=params['DATASET_PATH'],
        hr_filename=params['LABEL_HR_FILENAME'],
        exclude_interventions=True,
    )

    df_training = df_experiment[df_experiment['train_val_test'] == 'train'].sort_values(by=['exp_idx'])
    df_validation = df_experiment[df_experiment['train_val_test'] == 'val'].sort_values(by=['exp_idx'])
    df_test = df_experiment[df_experiment['train_val_test'] == 'test'].sort_values(by=['exp_idx'])

    # Create Data Transforms
    if params['APPLY_DATA_AUGMENTATION']:
        composed_transforms = transforms.Compose([
            signal_transforms.RandomGaussianNoise(),
            signal_transforms.RandomSignalFlip(),
            signal_transforms.RandomSlopeNoise(),
            signal_transforms.RandomStepNoise(),
            signal_transforms.MinMaxNorm(),
        ])
    else:
        composed_transforms = None

    # Data Loader Class Object
    train_set = data_loader.DataLoaderRhythmNet(
        df_dataloader=df_training,
        n_gru_inputs=params['N_INPUTS_GRU'],
        transform=composed_transforms,
    )

    validation_set = data_loader.DataLoaderRhythmNet(
        df_dataloader=df_validation,
        n_gru_inputs=params['N_INPUTS_GRU'],
        transform=composed_transforms,
    )

    test_set = data_loader.DataLoaderRhythmNet(
        df_dataloader=df_test,
        n_gru_inputs=params['N_INPUTS_GRU'],
        transform=composed_transforms,
    )

    # Dataset
    train_dataloader = DataLoader(
        train_set,
        batch_size=params['BATCH_SIZE'],
        shuffle=True
    )

    validation_dataloader = DataLoader(
        validation_set,
        batch_size=params['BATCH_SIZE'],
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=params['BATCH_SIZE'],
        shuffle=False
    )

    return train_dataloader, validation_dataloader, test_dataloader


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    RUN_SINGLE_TRAIN = False

    if RUN_SINGLE_TRAIN:

        config_file.params['LR'] = 0
        config_file.params['LOSS_WEIGHT_REG'] = 0
        config_file.params['LOSS_WEIGHT_GRU'] = 0
        config_file.params['MODEL_WITH_IR'] = True
        config_file.params['WEIGHTS_PRETRAINED'] = False
        config_file.params['APPLY_DATA_AUGMENTATION'] = True
        SAVE_NAME_MODEL = 'teat.pt'

        # Test GPU
        if torch.cuda.is_available():
            print('GPU available... using GPU')
            torch.cuda.manual_seed_all(10)
        else:
            print("GPU not available, using CPU")

        # Load Dataset
        train_dataloader, validation_dataloader, test_dataloader = load_dataset(config_file.params)

        # Model
        model = rythmnet.RhythmNet(
            pretrained=config_file.params['WEIGHTS_PRETRAINED'],
            ir_channel=config_file.params['MODEL_WITH_IR'],
        )
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=config_file.params['LR'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.8,
            patience=config_file.params['PATIENCE'],
            verbose=True
        )
        # Loss function
        # loss_fn = nn.L1Loss()
        loss_fn = loss_func.RhythmNetLoss(
            weight_reg=config_file.params['LOSS_WEIGHT_REG'],
            weight_gru=config_file.params['LOSS_WEIGHT_GRU'],
            gru_output=config_file.params['GRU_OUTPUT'],
        )

        # Parallelize
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)

        # --------------------------------------
        # Training
        # --------------------------------------

        train_loss_per_epoch = []
        train_metrics_per_epoch = {
            'MAE': [],
            'RMSE': [],
            'PEARSON_R': [],
        }
        val_loss_per_epoch = []
        val_metrics_per_epoch = {
            'MAE': [],
            'RMSE': [],
            'PEARSON_R': [],
        }

        for epoch in range(config_file.params['EPOCHS']):
            # Training
            model.train(True)
            target_hr_list, predicted_hr_list, train_loss = train_fn(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                params=config_file.params,
            )

            metrics_training = metrics.compute_criteria(target_hr_list, predicted_hr_list)

            for metric in metrics_training.keys():
                train_metrics_per_epoch[metric].append(metrics_training[metric])

            print(f"\nFinished [Epoch: {epoch + 1}/{config_file.params['EPOCHS']}]s",
                  "\nTraining Loss: {:.3f} |".format(train_loss),
                  "HR_MAE : {:.3f} |".format(metrics_training["MAE"]),
                  "HR_RMSE : {:.3f} |".format(metrics_training["RMSE"]),
                  "PEARSON_R : {:.3f} |".format(metrics_training["PEARSON_R"]),
                  )

            train_loss_per_epoch.append(train_loss)

            # Validation
            model.eval()
            val_target_hr_list, val_predicted_hr_list, val_loss = val_fn(
                model=model,
                val_dataloader=validation_dataloader,
                loss_fn=loss_fn,
                params=config_file.params,
            )

            metrics_validation = metrics.compute_criteria(val_target_hr_list, val_predicted_hr_list)

            for metric in metrics_validation.keys():
                val_metrics_per_epoch[metric].append(metrics_validation[metric])

            print(f"\nFinished [Epoch: {epoch + 1}/{config_file.params['EPOCHS']}]s",
                  "\nValidation Loss: {:.3f} |".format(val_loss),
                  "HR_MAE : {:.3f} |".format(metrics_validation["MAE"]),
                  "HR_RMSE : {:.3f} |".format(metrics_validation["RMSE"]),
                  "PEARSON_R : {:.3f} |".format(metrics_validation["PEARSON_R"]),
                  )

            val_loss_per_epoch.append(val_loss)

            # Save model with best test RMSE metric
            print(val_metrics_per_epoch)
            temp_val_rmse = metrics_validation['RMSE']
            if len(val_metrics_per_epoch['RMSE']) > 1:
                if temp_val_rmse < min_rsme:
                    print('saved model - Val RMSE', temp_val_rmse)
                    print('epoch', epoch)
                    save_model_checkpoint(
                        model,
                        optimizer,
                        train_loss,
                        config_file.params['CHECKPOINT_PATH'],
                        save_filename=SAVE_NAME_MODEL,
                    )
                    min_rsme = temp_val_rmse
            else:
                min_rsme = metrics_validation['RMSE']
                save_model_checkpoint(
                    model,
                    optimizer,
                    train_loss,
                    config_file.params['CHECKPOINT_PATH'],
                    save_filename=SAVE_NAME_MODEL,
                )
                print('saved model - Val RMSE', temp_val_rmse)
                print('epoch', epoch)

        # Training Plots
        fig = plt.figure(figsize=(10, 5))
        plt.plot(train_loss_per_epoch, c='orange', label='train')
        plt.plot(val_loss_per_epoch, c='green', label='val')
        plt.xlabel('Epochs')
        plt.ylabel('Train loss Per Epoch')
        plt.savefig('./results/figures/Loss_training.pdf')
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(10, 5))
        plt.plot(train_metrics_per_epoch["MAE"], c='orange', label='MAE-train')
        plt.plot(train_metrics_per_epoch["RMSE"], c='red', label='RMSE-train')
        plt.plot(val_metrics_per_epoch["MAE"], c='green', label='MAE-val')
        plt.plot(val_metrics_per_epoch["RMSE"], c='blue', label='RMSE-val')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics Per Epoch')
        plt.savefig('./results/figures/RMSE_training.pdf')
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(10, 5))
        plt.plot(train_metrics_per_epoch["PEARSON_R"], c='orange', label='R-train')
        plt.plot(val_metrics_per_epoch["PEARSON_R"], c='green', label='R-val')
        plt.xlabel('Epochs')
        plt.ylabel('PEARSON_R Per Epoch')
        plt.savefig('./results/figures/pearson_training.pdf')
        plt.legend()
        plt.show()
