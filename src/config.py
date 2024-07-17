params = {
    'PARTICIPANTS_TRAIN_VAL': ['1', '2', '3', '4'],
    'PARTICIPANTS_TEST': ['5'],
    'RATIO_DATA_TRAIN_VAL': [0.7, 0.3],
    'EXPERIMENT_NAME': '',
    'DATASET_PATH': './data/processed/',
    'CHANNELS': ['_y', '_cr', '_cb'],
    'MODEL_WITH_IR': True,
    'APPLY_DATA_AUGMENTATION': True,
    'LABEL_HR_FILENAME': 'hr_st_maps.csv',
    'N_INPUTS_GRU': 6,
    'BATCH_SIZE': 8,
    'PATIENCE': 5,
    'EPOCHS': 15,
    'N_TRAIN_EXAMPLES': 1000,  # 2346
    'N_VAL_EXAMPLES': 400,     # 1007
    'N_TEST_EXAMPLES': 400,    # 895
    'GRU_OUTPUT': True,
    'DEVICE': 'cuda',  # 'cuda'  or 'cpu
    'PROJECT_NAME_WANDB': '1_stmaps',
    'PROJECT_NAME_OPTUNA': '1_stmaps',
    'OPTUNA_METRIC': 'RMSE',
    'OPTUNA_N_TRIALS': 30,
    'CHECKPOINT_PATH': './models/rythmnet/checkpoint_1/',
    'SAVE_MODEL': True,
}

