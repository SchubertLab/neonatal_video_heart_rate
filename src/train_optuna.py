import optuna
import wandb
import torch

import src.config as config_file
import src.utils.metrics as metrics
import src.model.rythmnet as rythmnet
import src.utils.loss_func as loss_func
import src.train as train


def objective(trial, params):
    # Optuna and WandB configuration ----------------------------------------------------------------------------
    # Parameters to be optimized with the trial
    if params['MODEL_WITH_IR']:
        PRETRAINED_RESNET = False
    else:
        PRETRAINED_RESNET = trial.suggest_categorical(
            "pretrained_resnet",
            [True, False]
        )
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)
    LOSS_WEIGHT_REG = trial.suggest_float("weight_loss_regression", 1e-1, 1e3, log=True)
    LOSS_WEIGHT_GRU = trial.suggest_float("weight_loss_gru", 1e-1, 1e3, log=True)

    params['PRETRAINED_RESNET'] = PRETRAINED_RESNET
    params['LEARNING_RATE'] = LEARNING_RATE
    params['LOSS_WEIGHT_REG'] = LOSS_WEIGHT_REG
    params['LOSS_WEIGHT_GRU'] = LOSS_WEIGHT_GRU

    # WandB configuration
    PROJECT_NAME_WANDB = params['PROJECT_NAME_WANDB']
    run = wandb.init(
        project=PROJECT_NAME_WANDB,
        config=params,
    )
    config_wandb = wandb.config
    print('wandb', str(run.id), str(run.name))

    # Dataset, Model and Optimizer ----------------------------------------------------------------------------
    # Load Dataset
    train_dataloader, validation_dataloader, test_dataloader = train.load_dataset(params)

    # Model
    model = rythmnet.RhythmNet(
        pretrained=PRETRAINED_RESNET
    )

   # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=params['PATIENCE'],
        verbose=True
    )
    # Loss function
    loss_fn = loss_func.RhythmNetLoss(
        weight_reg=LOSS_WEIGHT_REG,
        weight_gru=LOSS_WEIGHT_GRU,
        gru_output=params['GRU_OUTPUT'],
    )

    # Parallelize
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Train and Validation   ----------------------------------------------------------------------------

    # Initialize Metrics per Epoch
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    metrics_per_epoch = {
        'MAE': [],
        'RMSE': [],
        'PEARSON_R': [],
    }
    train_metrics_per_epoch = metrics_per_epoch.copy()
    val_metrics_per_epoch = metrics_per_epoch.copy()

    # Log Model Training to WandB
    wandb.watch(model, log_freq=100)

    for epoch in range(params['EPOCHS']):
        # Model Training
        model.train(True)
        target_hr_list, predicted_hr_list, train_loss = train.train_fn(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            params=params,
        )
        metrics_training = metrics.compute_criteria(target_hr_list, predicted_hr_list)

        # Log training results to Wandb
        wandb.log({"epoch": epoch})
        wandb.log({"train_loss": train_loss})
        wandb.log({"train_mae": metrics_training["MAE"]})
        wandb.log({"train_rmse": metrics_training["RMSE"]})
        wandb.log({"train_pearson_r": metrics_training["PEARSON_R"]})

        for metric in metrics_training.keys():
            train_metrics_per_epoch[metric].append(metrics_training[metric])

        print(f"\nFinished [Epoch: {epoch + 1}/{params['EPOCHS']}]s",
              "\nTraining Loss: {:.3f} |".format(train_loss),
              "HR_MAE : {:.3f} |".format(metrics_training["MAE"]),
              "HR_RMSE : {:.3f} |".format(metrics_training["RMSE"]),
              "PEARSON_R : {:.3f} |".format(metrics_training["PEARSON_R"]),
              )

        train_loss_per_epoch.append(train_loss)

        # Validation per epoch
        model.eval()
        val_target_hr_list, val_predicted_hr_list, val_loss = train.val_fn(
            model=model,
            val_dataloader=validation_dataloader,
            loss_fn=loss_fn,
            params=params,
        )

        metrics_validation = metrics.compute_criteria(val_target_hr_list, val_predicted_hr_list)

        # Log validation results to Wandb
        wandb.log({"val_loss": train_loss})
        wandb.log({"val_mae": metrics_validation["MAE"]})
        wandb.log({"val_rmse": metrics_validation["RMSE"]})
        wandb.log({"val_pearson_r": metrics_validation["PEARSON_R"]})

        for metric in metrics_validation.keys():
            val_metrics_per_epoch[metric].append(metrics_validation[metric])

        print(f"\nFinished [Epoch: {epoch + 1}/{params['EPOCHS']}]s",
              "\nValidation Loss: {:.3f} |".format(val_loss),
              "HR_MAE : {:.3f} |".format(metrics_validation["MAE"]),
              "HR_RMSE : {:.3f} |".format(metrics_validation["RMSE"]),
              "PEARSON_R : {:.3f} |".format(metrics_validation["PEARSON_R"]),
              )

        val_loss_per_epoch.append(val_loss)
        trial.report(metrics_validation["RMSE"], epoch+1)
        trial.report(metrics_validation["PEARSON_R"], epoch + 1)

        # Save model with best loss
        if len(val_loss_per_epoch) > 0:
            if val_loss <= min(val_loss_per_epoch):
                if params['SAVE_MODEL']:
                    train.save_model_checkpoint(model, optimizer, train_loss, params['CHECKPOINT_PATH'])
                # temp_best_model = model.copy()
                temp_best_metrics = metrics_validation.copy()
        else:
            if params['SAVE_MODEL']:
                train.save_model_checkpoint(model, optimizer, train_loss, params['CHECKPOINT_PATH'])
            # temp_best_model = model.copy()
            temp_best_metrics = metrics_validation.copy()

    wandb.finish()

    # Report best score to Optuna
    print('Optuna - Val')
    print('temp_best_metrics', temp_best_metrics)
    print('MAE', temp_best_metrics['MAE'])
    print('RMSE', temp_best_metrics['RMSE'])
    print('PEARSON_R', temp_best_metrics['PEARSON_R'])

    metric_name = params['OPTUNA_METRIC']
    score_optuna = temp_best_metrics[metric_name]
    return score_optuna


if __name__ == "__main__":
    torch.cuda.manual_seed_all(10)
    torch.manual_seed(10)

    SAVE_MODEL = False
    N_TRIALS = config_file.params['OPTUNA_N_TRIALS']
    OPTUNA_STUDY_NAME = config_file.params['PROJECT_NAME_OPTUNA']

    config_parameters = config_file.params
    # ------------------------------------------------------------------------------------------
    # Create optuna study
    storage_name = "sqlite:///{}.db".format(OPTUNA_STUDY_NAME)
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(
        lambda trial: objective(
            trial,
            params=config_parameters,
        ),
        n_trials=N_TRIALS
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df_optimization = study.trials_dataframe()
    df_optimization.to_csv('data/optuna_trials/' + OPTUNA_STUDY_NAME + '.csv')
