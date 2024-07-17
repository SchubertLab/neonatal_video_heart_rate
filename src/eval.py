import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import src.config as config_file
import src.utils.metrics as metrics
import src.model.rythmnet as rythmnet
import src.dataset.data_loader as data_loader


def eval_fn(model, test_dataloader, params):
    model.eval()
    gru_outputs_total_array = []
    labels_total_array = []

    for batch_idx, batch_data in enumerate(test_dataloader):
        # Limiting training data for faster epochs.
        if batch_idx * params['BATCH_SIZE'] >= params['N_TEST_EXAMPLES']:
            break

        print('test_batch_idx', batch_idx)
        st_maps_batch = batch_data['st_maps']
        labels_batch = batch_data['target']

        with torch.no_grad():
            for batch_n in range(labels_batch.shape[0]):
                # Every data instance is an input + label pair
                inputs = st_maps_batch[batch_n, :, :, :, :]
                labels = labels_batch[batch_n, :]

                # Make predictions for this batch
                reg_outputs, gru_outputs = model(inputs)

                gru_outputs_total_array.append(gru_outputs.mean())
                labels_total_array.append(labels.mean())

    gru_outputs_total_array = [x.detach().numpy() for x in gru_outputs_total_array]
    gru_outputs_total_array = np.array(gru_outputs_total_array).flatten()

    labels_total_array = [x.detach().numpy() for x in labels_total_array]
    labels_total_array = np.array(labels_total_array).flatten()

    results = {
        'model_outputs': gru_outputs_total_array,
        'labels': labels_total_array,
    }

    eval_metrics = {
        'MAE': [],
        'RMSE': [],
        'PEARSON_R': [],
    }
    metrics_validation = metrics.compute_criteria(
        gru_outputs_total_array,
        labels_total_array,
    )

    for metric in eval_metrics.keys():
        eval_metrics[metric].append(metrics_validation[metric])

    return results, eval_metrics


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    RUN_SINGLE_TEST = False

    if RUN_SINGLE_TEST:

        MKT_TEST = '1'

        params = config_file.params
        params['N_TEST_EXAMPLES'] = 1000

        df_experiment = data_loader.build_dataframe_experiment(
            list_participants=params['PARTICIPANTS_TRAIN_VAL'],
            ratio_train_val_test=params['RATIO_DATA_TRAIN_VAL_TEST'],
            dataset_path=params['DATASET_PATH'],
            hr_filename=params['LABEL_HR_FILENAME'],
            st_map_file_ext='.npy'
        )

        df_test = df_experiment[df_experiment['train_val_test'] == 'test'].sort_values(by=['exp_idx'])
        df_test_mkt = df_test[df_test['mkt'] == MKT_TEST]
        print(len(df_test))

        evaluation_set = data_loader.DataLoaderRhythmNet(
            df_dataloader=df_test_mkt,
            n_gru_inputs=params['N_INPUTS_GRU'],
        )

        # Create evaluation dataloader.
        evaluation_dataloader = DataLoader(
            evaluation_set,
            batch_size=params['BATCH_SIZE'],
            shuffle=False
        )

        # Load saved model
        model = rythmnet.RhythmNet(
            pretrained=False
        )
        model_path = params['CHECKPOINT_PATH'] + 'running_model.pt'
        checkpoint = torch.load(model_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        eval_metrics = model.eval()

        # Model Predictions
        eval_target_hr_list = []
        eval_predicted_hr_list = []

        results_array, eval_metrics = eval_fn(model, evaluation_dataloader, params)

        print(eval_metrics)

        # Figures ---------------------------------------------------------------------------------
        # Fix arrays
        eval_predicted_hr_list = results_array['model_outputs']
        eval_target_hr_list = results_array['labels']

        min_max_ref = np.abs(np.max(eval_target_hr_list) - np.min(eval_target_hr_list))
        min_max_pred = np.abs(np.max(eval_predicted_hr_list) - np.min(eval_predicted_hr_list))

        norm_sig_ref = (eval_target_hr_list - np.min(eval_target_hr_list)) / min_max_ref
        norm_sig_pred = (eval_predicted_hr_list - np.min(eval_predicted_hr_list)) / min_max_pred

        fig = plt.figure(figsize=(10, 4))
        plt.plot(norm_sig_ref, marker='.', label='Ground Truth HR - ECG')
        plt.scatter(np.arange(len(norm_sig_pred)), norm_sig_pred, marker='.', c='orange',
                    label='Rythmnet Predicted HR')
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Normalized HR')
        plt.title('Rythmnet Validation')
        plt.show()

        fig = plt.figure(figsize=(10, 4))
        plt.plot(eval_target_hr_list, marker='.', label='Ground Truth HR - ECG')
        plt.scatter(np.arange(len(eval_predicted_hr_list)), eval_predicted_hr_list, marker='.', c='orange',
                    label='Rythmnet Predicted HR')
        plt.show()






